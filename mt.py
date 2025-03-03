!pip install transformers datasets evaluate sacrebleu tf-keras ipywidgets sentencepiece sacremoses "accelerate>=0.26.0"

!pip install nltk bert_score

# import libraries
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer, EarlyStoppingCallback)

# load dataset
dataset = load_dataset("qanastek/ELRC-Medical-V2", "en-fr")

# Create a proper train/test split to avoid contamination
ds = dataset["train"].train_test_split(test_size=0.2, seed=42) if "train" in dataset else dataset.train_test_split(test_size=0.2, seed=42)

# load pretrained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)  # Create separate base model for comparison

def tokenize_fn(examples):
    en_texts = [ex["en"] for ex in examples["translation"]]
    fr_texts = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(en_texts, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(fr_texts, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs
tokenized_ds = ds.map(tokenize_fn, batched=True)

# data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# training arguments with early stopping
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    save_strategy="epoch",  # Add this line to match evaluation_strategy
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=5,
    num_train_epochs=10,
    predict_with_generate=True,
    fp16=True,
    logging_dir="./logs",
    load_best_model_at_end=True,  # Enable loading the best model at the end of training
    metric_for_best_model="eval_bleu",  # Use BLEU for early stopping
    greater_is_better=True,  # Higher BLEU is better
)

# load multiple metrics
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

# setup trainer for fine-tuning with early stopping
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]  # Add early stopping
)

# fine-tune the model
trainer.train()

# evaluate the fine-tuned model on the test set
print("Evaluating fine-tuned model...")
eval_results = trainer.evaluate()
print("Fine-tuned model metrics:", eval_results)

# get a few original examples from the test set (for english and reference french)
sample_count = 5
sample_original = ds["test"].select(range(sample_count))
sample_tokenized = tokenized_ds["test"].select(range(sample_count))

# get predictions from the fine-tuned model
ft_predictions = trainer.predict(sample_tokenized)
ft_decoded_preds = tokenizer.batch_decode(ft_predictions.predictions, skip_special_tokens=True)

# get predictions from the base model
base_predictions = base_trainer.predict(sample_tokenized)
base_decoded_preds = tokenizer.batch_decode(base_predictions.predictions, skip_special_tokens=True)

# print comparisons: original english, fine-tuned pred, base model pred, and reference french
for i, example in enumerate(sample_original):
    english_text = example["translation"]["en"]
    ref_text = example["translation"]["fr"]
    print(f"example {i}:")
    print("english      :", english_text)
    print("sft prediction:", ft_decoded_preds[i])
    print("base prediction:", base_decoded_preds[i])
    print("reference    :", ref_text)
    print("-" * 50)