# Enhanced Machine Translation Model Training
# 1. Fine-tuning with Supervised Learning and W&B integration
# 2. Further improvement with Reinforcement Learning (PPO)

# Install required packages
# !pip install transformers datasets evaluate sacrebleu tf-keras ipywidgets sentencepiece sacremoses "accelerate>=0.26.0"
# !pip install nltk bert_score wandb trl peft bitsandbytes flash-attn deepspeed

# Import libraries
import evaluate
import numpy as np
import wandb
import torch
import os
from datasets import load_dataset
from transformers import (
    AutoTokenizer, 
    AutoModelForSeq2SeqLM,
    DataCollatorForSeq2Seq, 
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer, 
    EarlyStoppingCallback,
    TrainerCallback
)
import deepspeed

# Initialize wandb
wandb.login()  # You'll need to authenticate with your W&B account
wandb.init(project="medical-translation-sft-rl", name="en-fr-medical-translation")

# Load dataset
dataset = load_dataset("qanastek/ELRC-Medical-V2", "en-fr")

# Create a proper train/test split to avoid contamination
ds = dataset["train"].train_test_split(test_size=0.2, seed=42) if "train" in dataset else dataset.train_test_split(test_size=0.2, seed=42)

# Further split test into validation and test
test_valid = ds["test"].train_test_split(test_size=0.5, seed=42)
ds["validation"] = test_valid["train"]
ds["test"] = test_valid["test"]

# Load pretrained model and tokenizer with A100 optimizations
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Use BF16 mixed precision for A100s (more efficient than FP16 on A100)
model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",  # Automatically handle model parallelism if needed
    torch_dtype=torch.bfloat16  # Use BF16 precision (A100 has hardware acceleration for this)
)

base_model = AutoModelForSeq2SeqLM.from_pretrained(
    model_name,
    device_map="auto",
    torch_dtype=torch.bfloat16
)

def tokenize_fn(examples):
    en_texts = [ex["en"] for ex in examples["translation"]]
    fr_texts = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(en_texts, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(fr_texts, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = ds.map(tokenize_fn, batched=True)

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Create DeepSpeed config for A100 optimization
ds_config = {
    "fp16": {
        "enabled": False  # Disable FP16 since we'll use BF16
    },
    "bf16": {
        "enabled": True  # Enable BF16 for A100
    },
    "zero_optimization": {
        "stage": 2,  # ZeRO stage 2 for good balance of memory and speed
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "contiguous_gradients": True,
        "overlap_comm": True
    },
    "gradient_accumulation_steps": 4,  # Increase effective batch size
    "train_batch_size": 128,  # Global batch size
    "train_micro_batch_size_per_gpu": 32,  # Per-GPU batch size (larger for A100)
    "scheduler": {
        "type": "WarmupDecayLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-5,
            "warmup_num_steps": 500
        }
    }
}

# Save DeepSpeed config
import json
with open('ds_config.json', 'w') as f:
    json.dump(ds_config, f)

# Training arguments with W&B integration and A100 optimizations
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="steps",
    eval_steps=500,  # More frequent evaluation
    save_strategy="steps",
    save_steps=500,
    learning_rate=3e-5,  # Slightly higher learning rate for faster convergence
    per_device_train_batch_size=32,  # Increased batch size for A100
    per_device_eval_batch_size=32,
    gradient_accumulation_steps=4,  # Increase effective batch size
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=False,  # Disable FP16 since we're using BF16
    bf16=True,  # Enable BF16 for A100
    logging_dir="./logs",
    logging_steps=100,  # More frequent logging
    load_best_model_at_end=True,
    metric_for_best_model="eval_bleu",
    greater_is_better=True,
    report_to="wandb",  # Enable W&B reporting
    deepspeed="ds_config.json",  # Enable DeepSpeed
    generation_max_length=128,
    optim="adamw_torch_fused",  # Use fused AdamW optimizer
)

# Load multiple metrics
bleu_metric = evaluate.load("sacrebleu")
meteor_metric = evaluate.load("meteor")
bertscore_metric = evaluate.load("bertscore")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Post-process text
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[lbl.strip()] for lbl in decoded_labels]
    
    # Calculate all metrics
    bleu_result = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    meteor_result = meteor_metric.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])
    bertscore_result = bertscore_metric.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels], lang="fr")
    
    # Return all metrics
    return {
        "bleu": bleu_result["score"],
        "meteor": meteor_result["meteor"],
        "bertscore_f1": np.mean(bertscore_result["f1"])
    }

# Log the dataset info to W&B
wandb.log({
    "dataset_name": "qanastek/ELRC-Medical-V2",
    "language_pair": "en-fr",
    "train_examples": len(tokenized_ds["train"]),
    "val_examples": len(tokenized_ds["validation"]),
    "test_examples": len(tokenized_ds["test"]),
})

# Evaluate base model before fine-tuning
base_trainer = Seq2SeqTrainer(
    model=base_model,
    args=training_args,
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("Evaluating base model before fine-tuning...")
base_eval_results = base_trainer.evaluate()
print("Base model metrics:", base_eval_results)

# Log base model metrics to W&B
wandb.log({
    "base_bleu": base_eval_results["eval_bleu"],
    "base_meteor": base_eval_results["eval_meteor"],
    "base_bertscore": base_eval_results["eval_bertscore_f1"]
})

# Create a custom data collator that can handle batching efficiently
class OptimizedDataCollator(DataCollatorForSeq2Seq):
    def __call__(self, features, return_tensors=None):
        # Optimize padding strategy for faster processing
        batch = super().__call__(features, return_tensors)
        
        # Apply sequence length optimization by finding minimum required padding
        max_source_len = max(len(x) for x in batch["input_ids"])
        max_target_len = max(len(x) for x in batch["labels"] if x is not None)
        
        batch["input_ids"] = batch["input_ids"][:, :max_source_len]
        batch["attention_mask"] = batch["attention_mask"][:, :max_source_len]
        
        if "labels" in batch:
            batch["labels"] = batch["labels"][:, :max_target_len]
            
        return batch

# Use optimized data collator
optimized_data_collator = OptimizedDataCollator(tokenizer, model=model)

# Setup trainer for fine-tuning with early stopping and A100 optimizations
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["validation"],
    tokenizer=tokenizer,
    data_collator=optimized_data_collator,
    compute_metrics=compute_metrics,
    callbacks=[
        EarlyStoppingCallback(early_stopping_patience=3),
        ProfileCallback()  # Add throughput and memory profiling
    ]
)

# Fine-tune the model with SFT
print("Starting supervised fine-tuning...")
trainer.train()

# Save the fine-tuned model
model_path = "./medical_translation_sft"
trainer.save_model(model_path)
tokenizer.save_pretrained(model_path)

# Evaluate the fine-tuned model on the test set
print("Evaluating fine-tuned model...")
eval_results = trainer.evaluate(tokenized_ds["test"])
print("Fine-tuned model metrics:", eval_results)

# Log SFT model metrics to W&B
wandb.log({
    "sft_bleu": eval_results["eval_bleu"],
    "sft_meteor": eval_results["eval_meteor"],
    "sft_bertscore": eval_results["eval_bertscore_f1"]
})

# Get a few examples from the test set
sample_count = 5
sample_original = ds["test"].select(range(sample_count))
sample_tokenized = tokenized_ds["test"].select(range(sample_count))

# Get predictions from the fine-tuned model
ft_predictions = trainer.predict(sample_tokenized)
ft_decoded_preds = tokenizer.batch_decode(ft_predictions.predictions, skip_special_tokens=True)

# Get predictions from the base model
base_predictions = base_trainer.predict(sample_tokenized)
base_decoded_preds = tokenizer.batch_decode(base_predictions.predictions, skip_special_tokens=True)

# Print comparisons
for i, example in enumerate(sample_original):
    english_text = example["translation"]["en"]
    ref_text = example["translation"]["fr"]
    print(f"Example {i}:")
    print("English      :", english_text)
    print("SFT prediction:", ft_decoded_preds[i])
    print("Base prediction:", base_decoded_preds[i])
    print("Reference    :", ref_text)
    print("-" * 50)
    
    # Log examples to W&B
    wandb.log({
        f"example_{i}": wandb.Table(
            columns=["English", "SFT Prediction", "Base Prediction", "Reference"],
            data=[[english_text, ft_decoded_preds[i], base_decoded_preds[i], ref_text]]
        )
    })

# Create a comparison table in W&B
comparison_table = wandb.Table(
    columns=["Example", "English", "SFT Prediction", "Base Prediction", "Reference"]
)

for i, example in enumerate(sample_original):
    english_text = example["translation"]["en"]
    ref_text = example["translation"]["fr"]
    comparison_table.add_data(
        i, english_text, ft_decoded_preds[i], base_decoded_preds[i], ref_text
    )

wandb.log({"translation_comparison": comparison_table})

# ========== PART 2: Reinforcement Learning Fine-Tuning ==========

# Custom callback to profile memory usage and throughput
class ProfileCallback(TrainerCallback):
    def __init__(self):
        self.start_time = None
        self.step_count = 0
        self.total_samples = 0
    
    def on_train_begin(self, args, state, control, **kwargs):
        self.start_time = time.time()
    
    def on_step_end(self, args, state, control, **kwargs):
        self.step_count += 1
        self.total_samples += args.per_device_train_batch_size * args.gradient_accumulation_steps
        
        if self.step_count % 50 == 0:
            elapsed = time.time() - self.start_time
            samples_per_second = self.total_samples / elapsed
            
            # Get GPU memory stats
            if torch.cuda.is_available():
                allocated = torch.cuda.memory_allocated() / (1024 ** 3)
                max_allocated = torch.cuda.max_memory_allocated() / (1024 ** 3)
                
                wandb.log({
                    "throughput/samples_per_second": samples_per_second,
                    "memory/gpu_allocated_gb": allocated,
                    "memory/gpu_max_allocated_gb": max_allocated
                })
                
                print(f"Step {self.step_count}: {samples_per_second:.2f} samples/sec, "
                      f"GPU Memory: {allocated:.2f} GB (max: {max_allocated:.2f} GB)")

from trl import PPOTrainer, PPOConfig, AutoModelForSeq2SeqLMWithValueHead
from trl.core import LengthSampler
from peft import LoraConfig, PeftModel, PeftConfig, get_peft_model
import time

# Define a reward function based on BLEU score
def compute_reward(predictions, references):
    # Process predictions and references to match BLEU format
    processed_preds = [pred.strip() for pred in predictions]
    processed_refs = [[ref.strip()] for ref in references]
    
    # Calculate BLEU score
    bleu_result = bleu_metric.compute(predictions=processed_preds, references=processed_refs)
    
    # Normalize to a reasonable range for RL training
    return bleu_result["score"] / 100.0  # BLEU is 0-100, we scale to 0-1

# Initialize a new wandb run for RL phase
wandb.finish()  # End the SFT run
wandb.init(project="medical-translation-sft-rl", name="en-fr-medical-translation-rl")

# Load the fine-tuned model for RL
model = AutoModelForSeq2SeqLMWithValueHead.from_pretrained(model_path)

# Apply LoRA with A100-optimized parameters for parameter-efficient fine-tuning
peft_config = LoraConfig(
    task_type="SEQ_2_SEQ_LM",
    r=16,  # Increased rank for better capacity
    lora_alpha=32,
    lora_dropout=0.05,  # Reduced dropout for faster training
    target_modules=["q_proj", "v_proj", "k_proj", "out_proj", "fc1", "fc2"],  # Target more modules
    bias="none",  # Don't add bias parameters
    modules_to_save=["encoder.embed_tokens", "decoder.embed_tokens"],  # Save embedding layers
)
model = get_peft_model(model, peft_config)

# Add Flash Attention if available (significantly faster attention computation on A100)
try:
    from flash_attn import flash_attn_func, flash_attn_varlen_func
    print("Flash Attention is available and will be used!")
    # Flash Attention is automatically used if available in the transformers library
except ImportError:
    print("Flash Attention not available, using standard attention")

# PPO configuration optimized for A100
ppo_config = PPOConfig(
    learning_rate=2e-5,  # Increased learning rate for A100
    batch_size=64,  # Increased batch size for A100
    mini_batch_size=8,  # Increased for A100
    gradient_accumulation_steps=2,
    optimize_cuda_cache=True,
    early_stopping=True,
    target_kl=0.1,
    ppo_epochs=4,
    seed=42,
    log_with="wandb",
    bf16=True,  # Use BF16 precision on A100
    use_score_scaling=True,  # Add reward scaling for better stability
    score_clip=None,  # Don't clip rewards unnecessarily
    hook_to_accelerator=True  # Optimize with accelerate
)

# With A100, we can use larger datasets for RL fine-tuning
rl_train_dataset = tokenized_ds["train"].select(range(min(5000, len(tokenized_ds["train"]))))  # 5x larger
rl_eval_dataset = tokenized_ds["validation"].select(range(min(500, len(tokenized_ds["validation"]))))  # 5x larger

# Setup PPO trainer
ppo_trainer = PPOTrainer(
    config=ppo_config,
    model=model,
    tokenizer=tokenizer,
)

# Length sampler for generation
length_sampler = LengthSampler(min_length=8, max_length=128)

# Prepare dataloader
from torch.utils.data import DataLoader

def collate_fn(batch):
    input_texts = []
    target_texts = []
    
    for example in batch:
        input_ids = example["input_ids"]
        label_ids = example["labels"]
        
        # Convert IDs back to text
        input_text = tokenizer.decode(input_ids, skip_special_tokens=True)
        target_text = tokenizer.decode(label_ids, skip_special_tokens=True)
        
        input_texts.append(input_text)
        target_texts.append(target_text)
    
    return input_texts, target_texts

train_dataloader = DataLoader(
    rl_train_dataset, 
    batch_size=ppo_config.batch_size, 
    shuffle=True, 
    collate_fn=collate_fn
)

# Set environment variables for better performance
os.environ["TOKENIZERS_PARALLELISM"] = "true"  # Enable parallel tokenization

# Add memory clearing function
def clear_gpu_memory():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
        print("GPU memory cleared and stats reset")

# Run RL training loop with A100 optimizations
print("Starting RL fine-tuning with A100 optimizations...")
clear_gpu_memory()  # Start with clean GPU memory
for epoch in range(5):  # We can do more epochs with A100
    for step, (input_texts, target_texts) in enumerate(train_dataloader):
        # Tokenize inputs
        query_tensors = tokenizer(
            input_texts, 
            return_tensors="pt", 
            padding=True, 
            truncation=True
        ).input_ids.to(model.device)
        
        # Generate responses from the model
        response_tensors = []
        for query in query_tensors:
            gen_len = length_sampler()
            response = ppo_trainer.generate(
                query.unsqueeze(0), 
                max_new_tokens=gen_len, 
                do_sample=True, 
                temperature=0.7
            )
            response_tensors.append(response.squeeze())
        
        # Decode responses
        batch_responses = [
            tokenizer.decode(r, skip_special_tokens=True) 
            for r in response_tensors
        ]
        
        # Calculate rewards based on BLEU score compared to reference
        rewards = [compute_reward([resp], [ref]) for resp, ref in zip(batch_responses, target_texts)]
        
        # Run PPO step
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)
        
        # Log metrics
        wandb.log({
            "epoch": epoch,
            "step": step,
            "ppo/mean_reward": stats["ppo/mean_reward"],
            "ppo/loss/policy": stats["ppo/loss/policy"],
            "ppo/loss/value": stats["ppo/loss/value"],
            "ppo/policy/entropy": stats["ppo/policy/entropy"],
            "ppo/policy/approxkl": stats["ppo/policy/approxkl"],
        })
        
        if step % 10 == 0:
            print(f"Epoch {epoch}, Step {step}, Mean Reward: {stats['ppo/mean_reward']}")
    
    # Evaluate after each epoch
    print(f"Evaluating after epoch {epoch}...")
    eval_metrics = {}
    
    # Sample from eval dataset
    eval_sample = rl_eval_dataset.select(range(min(20, len(rl_eval_dataset))))
    eval_inputs, eval_references = collate_fn(eval_sample)
    
    # Generate translations
    eval_responses = []
    for input_text in eval_inputs:
        input_ids = tokenizer(input_text, return_tensors="pt").input_ids.to(model.device)
        output_ids = ppo_trainer.generate(input_ids, max_new_tokens=128)
        prediction = tokenizer.decode(output_ids[0], skip_special_tokens=True)
        eval_responses.append(prediction)
    
    # Calculate metrics
    bleu_result = bleu_metric.compute(predictions=eval_responses, references=[[ref] for ref in eval_references])
    eval_metrics["eval_bleu"] = bleu_result["score"]
    
    # Log eval metrics
    wandb.log({
        "epoch": epoch,
        "eval_bleu": eval_metrics["eval_bleu"]
    })
    
    print(f"Epoch {epoch} eval BLEU: {eval_metrics['eval_bleu']}")

# Save the RL fine-tuned model
rl_model_path = "./medical_translation_rl"
ppo_trainer.save_pretrained(rl_model_path)

# Final evaluation on test set
from transformers import pipeline

# Load the RL model for evaluation
translator = pipeline(
    "translation", 
    model=rl_model_path,
    tokenizer=tokenizer,
    device=0 if model.device.type == "cuda" else -1
)

# Get examples from test set
test_examples = ds["test"].select(range(min(100, len(ds["test"]))))
test_inputs = [ex["translation"]["en"] for ex in test_examples]
test_references = [ex["translation"]["fr"] for ex in test_examples]

# Generate translations with the RL model
rl_translations = translator(test_inputs, max_length=128)
rl_predictions = [t["translation_text"] for t in rl_translations]

# Calculate final metrics
final_bleu = bleu_metric.compute(
    predictions=rl_predictions, 
    references=[[ref] for ref in test_references]
)

final_meteor = meteor_metric.compute(
    predictions=rl_predictions, 
    references=test_references
)

final_bertscore = bertscore_metric.compute(
    predictions=rl_predictions, 
    references=test_references, 
    lang="fr"
)

# Log final results
final_results = {
    "rl_bleu": final_bleu["score"],
    "rl_meteor": final_meteor["meteor"],
    "rl_bertscore": np.mean(final_bertscore["f1"]),
}

wandb.log(final_results)

print("Final RL Model Metrics:")
print(final_results)

# Create a final comparison table
final_comparison = wandb.Table(
    columns=["Example", "English", "SFT Prediction", "RL Prediction", "Reference"]
)

# Load SFT model predictions for the same examples
sft_translator = pipeline(
    "translation", 
    model=model_path,
    tokenizer=tokenizer,
    device=0 if model.device.type == "cuda" else -1
)
sft_translations = sft_translator(test_inputs[:10], max_length=128)
sft_predictions = [t["translation_text"] for t in sft_translations]

for i in range(10):  # Show first 10 examples
    final_comparison.add_data(
        i, 
        test_inputs[i], 
        sft_predictions[i], 
        rl_predictions[i], 
        test_references[i]
    )

wandb.log({"final_translation_comparison": final_comparison})

# End W&B run
wandb.finish()