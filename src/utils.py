"""Utility functions for model saving, loading, and ONNX export"""

import torch
from transformers import AutoTokenizer


def get_model_size(model):
    """Get model size in MB.

    Args:
        model: PyTorch model

    Returns:
        float: Model size in megabytes
    """
    param_size = 0
    for param in model.parameters():
        param_size += param.nelement() * param.element_size()
    return param_size / (1024 * 1024)


def save_quantized_model(
    model,
    tokenizer,
    model_path="quantized_model.pt",
    tokenizer_dir="quantized_model_tokenizer",
):
    """Save quantized model and tokenizer to disk.

    Args:
        model: Quantized PyTorch model
        tokenizer: HuggingFace tokenizer
        model_path (str): Path to save model weights
        tokenizer_dir (str): Directory to save tokenizer

    Returns:
        tuple: (model_path, tokenizer_dir)
    """
    # Save the entire quantized model (not just state_dict)
    torch.save(model, model_path)
    print(f"Quantized model saved to '{model_path}'")

    # Save tokenizer
    tokenizer.save_pretrained(tokenizer_dir)
    print(f"Tokenizer saved to '{tokenizer_dir}/'")

    return model_path, tokenizer_dir


def load_quantized_model(model_path, tokenizer_dir, device="cpu"):
    """Load a quantized model and tokenizer from disk.

    Args:
        model_path (str): Path to saved quantized model
        tokenizer_dir (str): Directory containing saved tokenizer
        device (str): Device to load model on

    Returns:
        tuple: (model, tokenizer)
    """

    # Set the quantization backend before loading
    torch.backends.quantized.engine = "fbgemm"

    # Load tokenizer
    print(f"Loading tokenizer from '{tokenizer_dir}'...")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    # Load the quantized model
    print(f"Loading quantized model from '{model_path}'...")
    loaded_data = torch.load(model_path, map_location=device, weights_only=False)

    # Check if it's a state_dict or full model
    if isinstance(loaded_data, dict):
        # It's a state_dict, need to reconstruct the model
        # This won't work without the model architecture
        raise ValueError(
            "Loaded file is a state_dict. Please re-run main.py to save the full model. "
            "The save function has been updated to save the complete model object."
        )
    else:
        # It's the full model
        model = loaded_data

    model = model.to(device)
    model.eval()

    print(f"Model loaded with {sum(p.numel() for p in model.parameters())} parameters")

    return model, tokenizer


def export_to_onnx(
    model, tokenizer, onnx_path="quantized_model.onnx", dummy_text="Hello, how are you?"
):
    """Export a PyTorch model to ONNX format.

    Args:
        model: PyTorch model to export
        tokenizer: Tokenizer for creating dummy inputs
        onnx_path (str): Path to save ONNX model
        dummy_text (str): Sample text for tracing the model

    Returns:
        str: Path to exported ONNX model
    """
    print(f"Exporting model to ONNX format at '{onnx_path}'...")

    # Create dummy input
    inputs = tokenizer(dummy_text, return_tensors="pt")

    # Export to ONNX
    torch.onnx.export(
        model,
        (inputs["input_ids"], inputs["attention_mask"]),
        onnx_path,
        input_names=["input_ids", "attention_mask"],
        output_names=["logits"],
        dynamic_axes={
            "input_ids": {0: "batch_size", 1: "sequence_length"},
            "attention_mask": {0: "batch_size", 1: "sequence_length"},
            "logits": {0: "batch_size", 1: "sequence_length"},
        },
        opset_version=14,
    )

    print(f"Model exported to '{onnx_path}'")
    return onnx_path
