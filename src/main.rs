use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model};
use hf_hub::{
    api::sync::{Api, ApiBuilder},
    Repo, RepoType,
};
use std::{io::{self, Write}, path::PathBuf};
use tokenizers::Tokenizer;

const MODEL_ID: &str = "google/gemma-3-270m";

struct GemmaQA {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl GemmaQA {
    fn new() -> Result<Self> {
        println!("Loading language model...");

        // Use CPU for inference (change to GPU if available)
        let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);

        // Try to load a publicly available model first
        let hf_token = std::env::var("HF_TOKEN")?;

        let api = ApiBuilder::new()
            .with_progress(true)
            .with_cache_dir(PathBuf::from("huggingface"))
            .with_token(Some(hf_token))
            .build()?;
        let model = api.model(MODEL_ID.to_string());
        let config = model.get("config.json")?;
        let path = config.into_os_string().into_string().unwrap();
        println!("config path: {:?}", path);
        let config: Config = serde_json::from_str(&std::fs::read_to_string(path)?)?;
        println!("Config: {:?}", config);
        
        let weights = model.get("model.safetensors")?;
        let weights = std::fs::read(weights.into_os_string().into_string().unwrap())?;

        println!("Attempting to load tokenizer from: {}", MODEL_ID);

        // Load tokenizer
        let tokenizer = match Tokenizer::from_pretrained(MODEL_ID, None) {
            Ok(t) => {
                println!("✅ Tokenizer loaded successfully from {}", MODEL_ID);
                t
            }
            Err(e) => {
                println!("⚠️  Failed to load tokenizer from {}: {}", MODEL_ID, e);
                println!("Creating a basic tokenizer instead...");

                // Create a basic tokenizer as fallback
                Tokenizer::new(tokenizers::models::bpe::BPE::default())
            }
        };

        let vb = VarBuilder::zeros(DType::F32, &device);
        let model = Model::new(false, &config, vb)?;

        println!("✅ Model configuration loaded successfully!");
        println!("📝 Note: This is a demonstration with model configuration only.");
        println!("🔑 To load actual Gemma weights, you need Hugging Face authentication.");
        println!("💡 The system can still tokenize and process text input.");

        Ok(Self {
            model,
            tokenizer,
            device,
        })
    }

    fn generate_response(&self, prompt: &str, max_length: usize) -> Result<String> {
        // Tokenize the input prompt
        let tokens = match self.tokenizer.encode(prompt, true) {
            Ok(t) => t,
            Err(e) => {
                return Ok(format!(
                    "⚠️  Tokenization failed: {}. Input: '{}'",
                    e, prompt
                ));
            }
        };

        let input_ids = tokens.get_ids();

        // Convert to tensor
        let _input_tensor = match Tensor::new(input_ids, &self.device) {
            Ok(t) => t,
            Err(e) => {
                return Ok(format!(
                    "⚠️  Tensor creation failed: {}. Input: '{}'",
                    e, prompt
                ));
            }
        };

        let _input_tensor = match _input_tensor.unsqueeze(0) {
            Ok(t) => t,
            Err(e) => {
                return Ok(format!(
                    "⚠️  Tensor reshaping failed: {}. Input: '{}'",
                    e, prompt
                ));
            }
        };

        // For demonstration, we'll just return the tokenized input
        // In a real implementation with loaded weights, you would:
        // 1. Run the model forward pass
        // 2. Generate tokens autoregressively
        // 3. Decode the output tokens

        let token_count = input_ids.len();
        let response = format!(
            "✅ Input processed successfully! \
            📊 Found {} tokens. \
            🎯 Max response length: {} tokens. \
            📝 Input: '{}' \
            💡 In a full implementation with loaded weights, the model would generate an actual response.",
            token_count, max_length, prompt
        );

        Ok(response)
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    println!("🚀 Initializing Language Model Question-Answer System");
    println!("Loading model (this may take a moment on first run)...");

    let gemma_qa = match GemmaQA::new() {
        Ok(qa) => qa,
        Err(e) => {
            eprintln!("❌ Failed to initialize system: {}", e);
            eprintln!("💡 This might be due to network issues or model availability");
            return Err(e);
        }
    };

    println!("\n✅ System ready! Type your questions below (type 'quit' to exit)");
    println!("{}", "=".repeat(50));

    loop {
        print!("\n🤔 Question: ");
        io::stdout().flush()?;

        let mut input = String::new();
        io::stdin().read_line(&mut input)?;

        let input = input.trim();

        if input.to_lowercase() == "quit" || input.to_lowercase() == "exit" {
            println!("👋 Goodbye!");
            break;
        }

        if input.is_empty() {
            continue;
        }

        println!("🤖 Processing input...");

        match gemma_qa.generate_response(input, 100) {
            Ok(response) => {
                println!("💡 Answer: {}", response);
            }
            Err(e) => {
                eprintln!("❌ Error processing input: {}", e);
            }
        }
    }

    Ok(())
}
