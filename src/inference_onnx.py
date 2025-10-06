"""Simple ONNX inference script for quantized model

This script loads a quantized PyTorch model, exports it to ONNX format,
and runs simple inference.
"""

import numpy as np
import onnx
import onnxruntime as ort

from utils import export_to_onnx, load_quantized_model


def run_onnx_inference(onnx_path, tokenizer, prompt="What is machine learning?"):
    """Run inference using ONNX Runtime."""
    print(f"\nRunning ONNX inference on prompt: '{prompt}'")

    # Create ONNX Runtime session
    session = ort.InferenceSession(onnx_path)

    # Tokenize input
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True)

    # Prepare inputs for ONNX Runtime
    ort_inputs = {
        "input_ids": inputs["input_ids"].numpy(),
        "attention_mask": inputs["attention_mask"].numpy(),
    }

    # Run inference
    print("Running inference...")
    outputs = session.run(None, ort_inputs)
    logits = outputs[0]

    # Get the predicted next token
    next_token_id = np.argmax(logits[0, -1, :])
    next_token = tokenizer.decode([next_token_id])

    print(f"Input: {prompt}")
    print(f"Next predicted token: '{next_token}'")
    print(f"Output shape: {logits.shape}")

    return logits


def main():
    print("=" * 60)
    print("ONNX Quantized Model Inference")
    print("=" * 60)

    # Load the quantized model and tokenizer using utils
    model, tokenizer = load_quantized_model(
        "quantized_model.pt",
        "quantized_model_tokenizer",
        device="cpu",
    )

    # Export to ONNX using utils
    onnx_path = export_to_onnx(model, tokenizer)

    # Verify ONNX model
    print("\nVerifying ONNX model...")
    onnx_model = onnx.load(onnx_path)
    onnx.checker.check_model(onnx_model)
    print("ONNX model is valid!")

    # Run inference
    test_prompts = [
        "What is machine learning?",
        "Define neural networks.",
        "What is Python?",
    ]

    for prompt in test_prompts:
        run_onnx_inference(onnx_path, tokenizer, prompt)
        print("-" * 60)

    print("\nInference completed successfully!")


if __name__ == "__main__":
    main()
