# tinyQAT

A minimal PyTorch implementation of Quantization-Aware Training (QAT) for large language models, demonstrated with the Gemma-3-270m-IT model.

## What is Quantization-Aware Training (QAT)?

Quantization-Aware Training (QAT) is a technique that simulates the effects of quantization during model training by inserting fake quantization operations into the forward pass. This allows the model to learn parameters that are more robust to the precision loss that occurs when converting to lower-bit representations (e.g., INT8).

**How QAT works:**
- During training, all computations remain in FP32 for numerical stability
- `FakeQuantize` modules are inserted to simulate INT8 quantization effects by clamping and rounding values
- The model learns to compensate for quantization errors during training
- After training, the model is converted to a true quantized INT8 model

**Key benefits:**
- **Higher accuracy**: Models trained with QAT typically maintain better accuracy than post-training quantization
- **Smaller model size**: Reduces model size by ~4x (FP32 → INT8)
- **Faster inference**: INT8 operations are faster on compatible hardware
- **Lower memory footprint**: Reduced memory requirements for deployment

## QAT vs. Post-Training Quantization (PTQ)

There are two main approaches to model quantization:

### Post-Training Quantization (PTQ)
- **When**: Applied after training is complete
- **How**: Directly converts a trained FP32 model to INT8 using calibration data
- **Pros**: Simple, fast, no retraining required
- **Cons**: Can result in significant accuracy degradation, especially for smaller models or aggressive quantization

### Quantization-Aware Training (QAT)
- **When**: Applied during training (or fine-tuning)
- **How**: Simulates quantization effects during training so the model learns to compensate
- **Pros**: Better accuracy retention, more robust to quantization errors
- **Cons**: Requires training/fine-tuning, more computationally expensive

**Rule of thumb**: Use PTQ for quick experiments or when accuracy loss is acceptable. Use QAT when you need to minimize accuracy degradation or are quantizing to aggressive bit-widths.

## What This Repository Does

This repository demonstrates:
1. Loading a pre-trained LLM (Gemma-3-270m-IT)
2. Configuring QAT with PyTorch's native `torch.ao.quantization` API
3. Fine-tuning the model with fake quantization operations
4. Converting to a fully quantized INT8 model
5. Comparing model sizes and measuring compression ratios

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/tinyQAT.git
cd tinyQAT

# Install dependencies
pip install -r requirements.txt
```

### Dependencies

- **torch** (>=2.0.0): PyTorch for model training and quantization
- **transformers** (>=4.35.0): HuggingFace transformers for model loading
- **tqdm** (>=4.65.0): Progress bars for training
- **numpy** (>=1.24.0): Numerical operations
- **onnx** (>=1.14.0): ONNX model export and validation
- **onnxruntime** (>=1.15.0): ONNX model inference

## Usage

### Running the QAT Script

```bash
python main.py
```

This will:
1. Load the Gemma-3-270m-IT model in FP32
2. Prepare it for quantization-aware training
3. Train for 2 epochs on simple QA examples
4. Convert to a quantized INT8 model
5. Save the quantized model and tokenizer
6. Display model sizes and compression ratio

### Expected Output

```
Using device: cpu
Loading google/gemma-3-270m-it...
Model loaded with 268098176 parameters
Training on 5 examples
Preparing model for quantization-aware training...
QAT model prepared with 268098176 parameters
Starting QAT training for 2 epochs...

Epoch 1: Average Loss = 2.8353
Epoch 2: Average Loss = 1.3583

Converting to quantized model...
Original model size: 1022.71 MB
Quantized model size: 640.21 MB
Compression ratio: 1.60x
Quantized model saved to 'quantized_model.pt'
Tokenizer saved to 'quantized_model_tokenizer/'
```

### Running ONNX Inference

After training, you can export the model to ONNX format and run inference:

```bash
python src/inference_onnx.py
```

This will:
1. Load the saved quantized model and tokenizer from `quantized_model.pt` and `quantized_model_tokenizer/`
2. Export the model to ONNX format (`quantized_model.onnx`)
3. Verify the ONNX model is valid
4. Run inference on sample prompts using ONNX Runtime

**Note:** ONNX export for quantized transformer models is under development.

## Project Structure

```
tinyQAT/
├── src/
│   ├── __init__.py
│   ├── utils.py              # Model save/load and ONNX export utilities
│   └── inference_onnx.py     # ONNX inference script
├── main.py                    # QAT training script
├── requirements.txt
└── README.md
```

## Implementation Details

- **QAT Configuration**: Defaults to `qnnpack` backend for quantization configuration (MPS)
- **Training Data**: Simple question-answer pairs for demonstration (replace with real data for production)
- **Model**: Gemma-3-270m-IT (270M parameters)
- **Precision**: FP32 during training (required for QAT), INT8 after conversion
- **Batch Size**: 1 (small for demonstration purposes)
- **Model Saving**: Saves the complete quantized model object (not just state_dict) for easy loading
- **Utils Module**: Centralized utilities for model I/O and ONNX operations

## Customization

To use QAT with your own model or data:

1. Replace `model_name` with your model
2. Update `train_texts` with your training data
3. Adjust hyperparameters (`num_epochs`, `batch_size`, `learning_rate`)
4. Modify save paths if needed (`quantized_model.pt`, `quantized_model_tokenizer/`)

## References

- [PyTorch Quantization Documentation](https://pytorch.org/docs/stable/quantization.html)
- [Quantization-Aware Training Blog Post](https://pytorch.org/blog/quantization-aware-training/)
- [Integer Quantization for Deep Learning Inference](https://arxiv.org/abs/2004.09602)
