use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_nn::VarBuilder;
use candle_transformers::models::gemma::{Config, Model};
use hf_hub::api::sync::ApiBuilder;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::io::Read;
use std::{
    path::PathBuf,
};
use anyhow::{Error as E};
use tokenizers::Tokenizer;
use candle_transformers::generation::LogitsProcessor;

mod token_output_stream;
use crate::token_output_stream::TokenOutputStream;

const MODEL_ID: &str = "google/gemma-3-270m";

struct TextGeneration {
    model: Model,
    device: Device,
    tokenizer: TokenOutputStream,
    logits_processor: LogitsProcessor,
    repeat_penalty: f32,
    repeat_last_n: usize,
}

impl TextGeneration {
    #[allow(clippy::too_many_arguments)]
    fn new(
        model: Model,
        tokenizer: Tokenizer,
        seed: u64,
        temp: Option<f64>,
        top_p: Option<f64>,
        repeat_penalty: f32,
        repeat_last_n: usize,
        device: &Device,
    ) -> Self {
        let logits_processor = LogitsProcessor::new(seed, temp, top_p);
        Self {
            model,
            tokenizer: TokenOutputStream::new(tokenizer),
            logits_processor,
            repeat_penalty,
            repeat_last_n,
            device: device.clone(),
        }
    }

    fn run(&mut self, prompt: &str, sample_len: usize) -> Result<()> {
        use std::io::Write;
        self.tokenizer.clear();
        let mut tokens = self
            .tokenizer
            .tokenizer()
            .encode(prompt, true)
            .map_err(E::msg)?
            .get_ids()
            .to_vec();
        for &t in tokens.iter() {
            if let Some(t) = self.tokenizer.next_token(t)? {
                print!("{t}")
            }
        }
        std::io::stdout().flush()?;

        let mut generated_tokens = 0usize;
        let eos_token = match self.tokenizer.get_token("<eos>") {
            Some(token) => token,
            None => anyhow::bail!("cannot find the <eos> token"),
        };

        let eot_token = match self.tokenizer.get_token("<end_of_turn>") {
            Some(token) => token,
            None => {
                println!(
                    "Warning: <end_of_turn> token not found in tokenizer, using <eos> as a backup"
                );
                eos_token
            }
        };

        let start_gen = std::time::Instant::now();
        for index in 0..sample_len {
            let context_size = if index > 0 { 1 } else { tokens.len() };
            let start_pos = tokens.len().saturating_sub(context_size);
            let ctxt = &tokens[start_pos..];
            let input = Tensor::new(ctxt, &self.device)?.unsqueeze(0)?;
            let logits = self.model.forward(&input, start_pos)?;
            let logits = logits.squeeze(0)?.squeeze(0)?.to_dtype(DType::F32)?;
            let logits = if self.repeat_penalty == 1. {
                logits
            } else {
                let start_at = tokens.len().saturating_sub(self.repeat_last_n);
                candle_transformers::utils::apply_repeat_penalty(
                    &logits,
                    self.repeat_penalty,
                    &tokens[start_at..],
                )?
            };

            let next_token = self.logits_processor.sample(&logits)?;
            tokens.push(next_token);
            generated_tokens += 1;
            if next_token == eos_token || next_token == eot_token {
                break;
            }
            if let Some(t) = self.tokenizer.next_token(next_token)? {
                print!("{t}");
                std::io::stdout().flush()?;
            }
        }
        let dt = start_gen.elapsed();
        if let Some(rest) = self.tokenizer.decode_rest().map_err(E::msg)? {
            print!("{rest}");
        }
        std::io::stdout().flush()?;
        println!(
            "\n{generated_tokens} tokens generated ({:.2} token/s)",
            generated_tokens as f64 / dt.as_secs_f64(),
        );
        Ok(())
    }
}

struct GemmaQA {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}
fn load_safetensors(path: &str, device: &Device) -> Result<HashMap<String, Tensor>> {
    let mut bytes = Vec::new();
    let mut file = File::open(path)?;
    file.read_to_end(&mut bytes)?;
    let safetensors = SafeTensors::deserialize(&bytes)?;

    let mut tensors = HashMap::new();
    for (name, view) in safetensors.tensors() {
        // Load each tensor onto the device
        let tensor = Tensor::from_slice(view.data(), view.shape(), device)?;
        tensors.insert(name.to_string(), tensor);
    }
    Ok(tensors)
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
        let config: Config = serde_json::from_str(&std::fs::read_to_string(path)?)?;

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

        let tensor_path = model.get("model.safetensors")?;
        let tensors = load_safetensors(
            &tensor_path.into_os_string().into_string().unwrap(),
            &device,
        )?;
        println!("Tensors: {:?}", tensors);
        let vb = VarBuilder::from_tensors(tensors, DType::F32, &device);
        let model = Model::new(false, &config, vb)?;
        println!("Model: {:?}", model);

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

    let mut pipeline = TextGeneration::new(gemma_qa.model, gemma_qa.tokenizer, 42, None, None, 1.0, 64, &gemma_qa.device);
    pipeline.run("Hello, how are you?", 100)?;


    Ok(())
}
