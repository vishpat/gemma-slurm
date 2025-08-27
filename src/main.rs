use anyhow::Error as E;
use anyhow::Result;
use candle_core::{DType, Device, Tensor};
use candle_examples::token_output_stream::TokenOutputStream;
use candle_nn::VarBuilder;
use candle_transformers::generation::LogitsProcessor;
use candle_transformers::models::gemma3::{Config, Model};
use hf_hub::api::sync::ApiBuilder;
use std::path::PathBuf;
use tokenizers::Tokenizer;

const MODEL_ID: &str = "google/gemma-3-1b-it";
const TEMPERATURE: f32 = 1.0;
const TOP_P: f32 = 0.95;
const REPEAT_PENALTY: f32 = 1.1;
const REPEAT_LAST_N: usize = 64;

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

struct GemmaModel {
    model: Model,
    tokenizer: Tokenizer,
    device: Device,
}

impl GemmaModel {
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
        println!("Loading config from: {}", path);

        let config: Config = serde_json::from_str(&std::fs::read_to_string(path)?)?;

        println!("Attempting to load tokenizer from: {}", MODEL_ID);

        let tokenizer_path = model.get("tokenizer.json")?;
        println!(
            "Loading tokenizer from: {}",
            tokenizer_path.into_os_string().into_string().unwrap()
        );

        // Load tokenizer
        let tokenizer = match Tokenizer::from_pretrained(MODEL_ID, None) {
            Ok(t) => {
                println!("✅ Tokenizer loaded successfully from {}", MODEL_ID);
                t
            }
            Err(e) => {
                anyhow::bail!("Failed to load tokenizer from {}: {}", MODEL_ID, e);
            }
        };

        let tensor_path = model.get("model.safetensors")?;
        println!(
            "Loading tensors from: {}",
            &tensor_path.clone().into_os_string().into_string().unwrap()
        );

        let dtype = if device.is_cuda() {
            DType::BF16
        } else {
            DType::F32
        };
        let vb = unsafe { VarBuilder::from_mmaped_safetensors(&[tensor_path], dtype, &device)? };
        let model = Model::new(false, &config, vb)?;

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

    let gemma_model = match GemmaModel::new() {
        Ok(model) => model,
        Err(e) => {
            eprintln!("❌ Failed to initialize system: {}", e);
            eprintln!("💡 This might be due to network issues or model availability");
            return Err(e);
        }
    };

    println!("\n✅ System ready! Type your questions below (type 'quit' to exit)");
    println!("{}", "=".repeat(50));

    let mut pipeline = TextGeneration::new(
        gemma_model.model,
        gemma_model.tokenizer,
        100,
        Some(TEMPERATURE.into()),
        Some(TOP_P.into()),
        REPEAT_PENALTY,
        REPEAT_LAST_N,
        &gemma_model.device,
    );
    let prompt = "Classify: This product is amazing";
    let prompt_it = format!(
        "<start_of_turn> user\n{}<end_of_turn>\n<start_of_turn> model\n",
        prompt
    );
    pipeline.run(&prompt_it, 100)?;

    Ok(())
}
