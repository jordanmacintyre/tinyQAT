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


def prepare_batch(texts, tokenizer, max_length=128, device="cpu"):
    """Tokenize and prepare batch for training."""
    encoded = tokenizer(
        texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt"
    )
    return {
        "input_ids": encoded["input_ids"].to(device),
        "attention_mask": encoded["attention_mask"].to(device),
        "labels": encoded["input_ids"].to(device),  # For causal LM
    }


def main():
    # Set device
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    print(f"Using device: {device}")

    # Load tokenizer and model
    model_name = "google/gemma-3-270m-it"
    print(f"Loading {model_name}...")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.float32,
        device_map=None,
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
    qconfig = get_default_qat_qconfig("qnnpack")  # Use fbgemm backend
    model.qconfig = qconfig

    # Prepare model for QAT
    print("Preparing model for quantization-aware training...")
    qat_model = prepare_qat(model, inplace=False)
    qat_model = qat_model.to(device)

    print(
        f"QAT model prepared with {sum(p.numel() for p in qat_model.parameters())} parameters"
    )

    # Training setup
    optimizer = optim.AdamW(qat_model.parameters(), lr=1e-5)
    num_epochs = 2
    batch_size = 1  # Small batch for demo

    print(f"Starting QAT training for {num_epochs} epochs...")

    training_losses = []

    for epoch in range(num_epochs):
        epoch_losses = []

        # Simple batching
        for i in tqdm(range(0, len(train_texts), batch_size), desc=f"Epoch {epoch+1}"):
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
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss:.4f}")

    print("Training completed!")

    # Convert to quantized model
    print("Converting to quantized model...")
    torch.backends.quantized.engine = "qnnpack"
    quantized_model = convert(qat_model.eval(), inplace=False)

    # Compare model sizes
    original_size = get_model_size(model)
    quantized_size = get_model_size(quantized_model)
    compression_ratio = original_size / quantized_size

    print(f"Original model size: {original_size:.2f} MB")
    print(f"Quantized model size: {quantized_size:.2f} MB")
    print(f"Compression ratio: {compression_ratio:.2f}x")

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
