"""Simple PyTorch QAT with Gemma-3-270m-IT

Minimal quantization-aware training setup using `torch.ao.quantization` with the Gemma-3-270m-IT model.
"""

import warnings

import torch
import torch.optim as optim
from torch.ao.quantization import convert, get_default_qat_qconfig, prepare_qat
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from utils import get_model_size, save_quantized_model

warnings.filterwarnings("ignore")


def prepare_batch(texts, tokenizer, max_length=128, device="cuda:1"):
    """Tokenize and prepare batch for causal language modeling."""
    encoded = tokenizer(
        texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )

    # Shift for next-token prediction
    input_ids = encoded["input_ids"]
    attention_mask = encoded["attention_mask"]
    labels = encoded["input_ids"].clone()

    # Set padding tokens to -100 so they're ignored in loss calculation
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        "input_ids": input_ids.to(device),
        "attention_mask": attention_mask.to(device),
        "labels": labels.to(device),
    }


def validate_quantization_accuracy(fp32_model, int8_model, test_data):
    """Proper quantization validation with comprehensive metrics"""
    import torch.nn.functional as F

    kl_divs = []
    top1_agreements = []

    # For perplexity: track total loss and total tokens
    fp32_total_loss = 0
    fp32_total_tokens = 0
    int8_total_loss = 0
    int8_total_tokens = 0

    fp32_model.eval()
    int8_model.eval()

    with torch.no_grad():
        for batch in test_data:
            inputs = batch["input_ids"]
            labels = batch["labels"]
            attention_mask = batch["attention_mask"]

            # Get outputs (includes loss and logits)
            fp32_outputs = fp32_model(
                input_ids=inputs, attention_mask=attention_mask, labels=labels
            )
            int8_outputs = int8_model(
                input_ids=inputs, attention_mask=attention_mask, labels=labels
            )

            fp32_logits = fp32_outputs.logits
            int8_logits = int8_outputs.logits

            # 1. KL Divergence (distribution shift)
            fp32_lp = F.log_softmax(fp32_logits, dim=-1)
            int8_lp = F.log_softmax(int8_logits, dim=-1)
            kl = F.kl_div(int8_lp, fp32_lp, reduction="batchmean", log_target=True)
            kl_divs.append(kl.item())

            # 2. Top-1 Agreement (prediction consistency)
            fp32_preds = torch.argmax(fp32_logits, dim=-1)
            int8_preds = torch.argmax(int8_logits, dim=-1)
            agreement = (fp32_preds == int8_preds).float().mean()
            top1_agreements.append(agreement.item())

            # 3. Perplexity (generation quality) - weighted by token count
            # Count valid tokens (excluding padding marked as -100)
            num_tokens = (labels != -100).sum().item()

            if num_tokens > 0:
                fp32_total_loss += fp32_outputs.loss.item() * num_tokens
                fp32_total_tokens += num_tokens

                int8_total_loss += int8_outputs.loss.item() * num_tokens
                int8_total_tokens += num_tokens

    # Calculate metrics
    avg_kl = sum(kl_divs) / len(kl_divs)
    avg_agreement = (sum(top1_agreements) / len(top1_agreements)) * 100

    # Calculate perplexity correctly (weighted average loss, then exp)
    avg_fp32_loss = (
        fp32_total_loss / fp32_total_tokens if fp32_total_tokens > 0 else float("inf")
    )
    avg_int8_loss = (
        int8_total_loss / int8_total_tokens if int8_total_tokens > 0 else float("inf")
    )

    avg_fp32_ppl = torch.exp(torch.tensor(avg_fp32_loss)).item()
    avg_int8_ppl = torch.exp(torch.tensor(avg_int8_loss)).item()
    ppl_degradation = ((avg_int8_ppl - avg_fp32_ppl) / avg_fp32_ppl) * 100

    # Print results with interpretation
    print("=" * 60)
    print("Quantization Validation Results")
    print("=" * 60)
    print(f"KL Divergence: {avg_kl:.4f}", end="")
    if avg_kl < 0.01:
        print(" (Excellent)")
    elif avg_kl < 0.05:
        print(" (Good)")
    elif avg_kl < 0.1:
        print(" (Acceptable)")
    else:
        print(" (Poor)")

    print(f"Top-1 Agreement: {avg_agreement:.1f}%", end="")
    if avg_agreement > 95:
        print(" (Excellent)")
    elif avg_agreement > 90:
        print(" (Good)")
    elif avg_agreement > 85:
        print(" (Acceptable)")
    else:
        print(" (Poor)")

    print(f"FP32 Perplexity: {avg_fp32_ppl:.2f}")
    print(f"INT8 Perplexity: {avg_int8_ppl:.2f}")
    print(f"Degradation: {ppl_degradation:+.1f}%", end="")
    if abs(ppl_degradation) < 2:
        print(" (Excellent)")
    elif abs(ppl_degradation) < 5:
        print(" (Good)")
    elif abs(ppl_degradation) < 10:
        print(" (Acceptable)")
    else:
        print(" (Poor)")

    print("=" * 60)

    return {
        "kl_divergence": avg_kl,
        "top1_agreement": avg_agreement,
        "perplexity_degradation": ppl_degradation,
    }


def main():
    # Set device
    device = "cuda:1"
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = "google/gemma-3-270m-it"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
        attn_implementation="eager",
    )

    # Add padding token if not present
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = model.to(device)
    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

    # Simple training data
    train_texts = [
        "What is machine learning? Machine learning is a subset of AI that enables computers to learn from data.",
        "Explain quantum computing. Quantum computing uses quantum mechanics to process information.",
        "What is Python? Python is a high-level programming language known for its simplicity.",
        "Define neural networks. Neural networks are computing systems inspired by biological neural networks.",
        "What is deep learning? Deep learning is a subset of machine learning using neural networks.",
    ]

    print(f"Training on {len(train_texts)} examples")

    # Setup QAT configuration
    model.train()

    # Configure quantization
    qconfig = get_default_qat_qconfig("fbgemm")  # Use fbgemm backend
    model.qconfig = qconfig

    # Prepare model for QAT
    print("Preparing model for quantization-aware training...")
    qat_model = prepare_qat(model, inplace=False)
    qat_model = qat_model.to(device)

    print(
        f"QAT model prepared with {sum(p.numel() for p in qat_model.parameters())} parameters"
    )

    # Training setup
    optimizer = optim.AdamW(qat_model.parameters(), lr=1e-4)
    num_epochs = 20
    batch_size = 1  # Small batch for demo

    print(f"Starting QAT training for {num_epochs} epochs...")

    training_losses = []

    # Create epoch-level progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Training Progress")

    for epoch in epoch_pbar:
        epoch_losses = []

        # Simple batching
        for i in range(0, len(train_texts), batch_size):
            batch_texts = train_texts[i : i + batch_size]

            # Prepare batch
            batch = prepare_batch(batch_texts, tokenizer, device=device)

            # Forward pass
            optimizer.zero_grad()
            outputs = qat_model(**batch)
            loss = outputs.loss

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(qat_model.parameters(), 1.0)
            optimizer.step()

            epoch_losses.append(loss.item())

        avg_loss = sum(epoch_losses) / len(epoch_losses)
        training_losses.append(avg_loss)

        # Update progress bar with loss info
        epoch_pbar.set_postfix({"Loss": f"{avg_loss:.4f}"})

    print("Training completed!")

    # Extract FP32 trained model without FakeQuant modules
    print("\nExtracting trained FP32 model (removing FakeQuant modules)...")
    from torch.ao.quantization import DeQuantStub

    # Create a clean FP32 model by loading the base architecture and copying trained weights
    fp32_trained_model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.float32,
        device_map=None,
    ).to("cpu")

    # Copy weights from QAT model to FP32 model (skip FakeQuant modules)
    qat_state_dict = qat_model.state_dict()
    fp32_state_dict = fp32_trained_model.state_dict()

    # Transfer matching weights
    for name, param in fp32_state_dict.items():
        if name in qat_state_dict:
            fp32_state_dict[name] = qat_state_dict[name]

    fp32_trained_model.load_state_dict(fp32_state_dict, strict=False)
    fp32_trained_model.eval()

    print("Trained FP32 model extracted successfully")

    # Validate quantization accuracy before full INT8 conversion
    print("\nValidating quantization accuracy...")
    print("Comparing trained FP32 model vs QAT model outputs (fake-quantized)...")
    test_data = [prepare_batch([text], tokenizer, device="cpu") for text in train_texts]

    # Move models to CPU for comparison
    qat_model_cpu = qat_model.to("cpu").eval()

    validate_quantization_accuracy(fp32_trained_model, qat_model_cpu, test_data)

    # Convert to quantized model
    print("\nConverting to INT8 quantized model...")
    torch.backends.quantized.engine = "fbgemm"
    quantized_model = convert(qat_model_cpu, inplace=False)

    # Compare model sizes
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    compression_ratio = original_size / quantized_size

    print(f"\nOriginal model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")

    # Save the trained FP32 model (without FakeQuant)
    print("\nSaving trained FP32 model...")
    fp32_trained_model.save_pretrained("trained_fp32_model")
    tokenizer.save_pretrained("trained_fp32_model")
    print("Trained FP32 model saved to 'trained_fp32_model/'")

    # Save the quantized model using utils
    save_quantized_model(quantized_model, tokenizer)

    print("\nScript completed successfully!")
    print(f"Loaded Gemma-3-270m-IT model")
    print(f"Applied quantization-aware training")
    print(f"Achieved {compression_ratio:.2f}x compression")
    print(
        f"Training loss reduced by {((training_losses[0] - training_losses[-1]) / training_losses[0] * 100):.2f}%"
    )


if __name__ == "__main__":
    main()
