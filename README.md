# Gemma 3 270M Question-Answer Console Application

A Rust-based console application that demonstrates loading and using language models with the Candle crate. The application is designed to work with Gemma 3 270M but currently demonstrates the framework with a basic tokenizer.

## Features

- 🚀 Demonstrates Candle crate integration for language models
- 💬 Interactive console-based question-answer interface
- 🧠 Uses the Candle crate for efficient model inference framework
- 🔄 Text tokenization and processing
- 🎯 Easy-to-use command-line interface
- 🛡️ Graceful fallback when models are unavailable

## Current Status

⚠️ **Important Note**: This is currently a demonstration application that shows the framework structure. The full Gemma 3 270M model loading requires:

1. **Hugging Face Authentication**: You need to be logged in to Hugging Face and have access to the Gemma models
2. **Model Weights**: The actual model weights (~500MB) need to be downloaded
3. **Proper Tokenizer**: The Gemma-specific tokenizer needs to be loaded

## What Works Now

- ✅ Application framework and structure
- ✅ Candle crate integration
- ✅ Basic text processing
- ✅ Console interface
- ✅ Error handling and fallbacks

## What Needs Implementation

- 🔄 Full model weight loading from Hugging Face
- 🔄 Proper tokenization for the specific model
- 🔄 Actual text generation/inference
- 🔄 Model state management

## Prerequisites

- Rust (latest stable version)
- Internet connection (for downloading models when implemented)

## Installation

1. Clone or navigate to this project directory
2. Install dependencies:
   ```bash
   cargo build
   ```

## Usage

1. Run the application:
   ```bash
   cargo run
   ```

2. The application will attempt to load a tokenizer and model configuration

3. Type your questions and press Enter

4. Type `quit` or `exit` to close the application

## Example Usage

```
🚀 Initializing Language Model Question-Answer System
Loading model (this may take a moment on first run)...
Loading language model...
Attempting to load tokenizer from: distilbert-base-uncased
⚠️  Failed to load tokenizer from distilbert-base-uncased: request error: status code 401
Creating a basic tokenizer instead...
✅ Model configuration loaded successfully!
📝 Note: This is a demonstration with model configuration only.
🔑 To load actual Gemma weights, you need Hugging Face authentication.
💡 The system can still tokenize and process text input.

✅ System ready! Type your questions below (type 'quit' to exit)
==================================================

🤔 Question: How old are you?
🤖 Processing input...
💡 Answer: ✅ Input processed successfully! 📊 Found 0 tokens...

🤔 Question: quit
👋 Goodbye!
```

## Technical Details

- **Framework**: Candle crate for Rust
- **Model**: Gemma 3 270M configuration (weights not loaded)
- **Inference**: CPU-based framework ready
- **Tokenization**: Basic BPE tokenizer (fallback)
- **Error Handling**: Graceful fallbacks and informative messages

## Next Steps for Full Implementation

To make this a fully functional Gemma 3 270M application:

1. **Authentication Setup**:
   ```bash
   # Set Hugging Face token
   export HF_TOKEN="your_token_here"
   ```

2. **Model Loading**: Implement proper weight downloading from Hugging Face

3. **Tokenizer**: Load the correct Gemma tokenizer

4. **Inference**: Implement the full text generation pipeline

## Dependencies

- `candle-core`: Core tensor operations
- `candle-transformers`: Transformer model implementations
- `candle-nn`: Neural network utilities
- `tokenizers`: Hugging Face tokenization (with http feature)
- `anyhow`: Error handling
- `tokio`: Async runtime

## Notes

- The application currently demonstrates the framework structure
- Model weights are not loaded due to authentication requirements
- Basic tokenization is available as a fallback
- The console interface is fully functional

## Troubleshooting

- **Authentication errors**: You need Hugging Face access for Gemma models
- **Model loading fails**: Check internet connection and Hugging Face availability
- **Basic functionality**: The app works for demonstration purposes
- **Full functionality**: Requires implementing weight loading and proper tokenization

## Contributing

This is a demonstration project. To contribute to making it fully functional:

1. Implement Hugging Face authentication
2. Add model weight downloading
3. Implement proper tokenization
4. Add full text generation pipeline
5. Improve error handling and user experience
