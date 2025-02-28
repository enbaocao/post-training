!pip install transformers datasets evaluate sacrebleu tf-keras ipywidgets sentencepiece sacremoses "accelerate>=0.26.0"

# import libraries
import evaluate
import numpy as np
from datasets import load_dataset
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Seq2SeqTrainingArguments,
                          Seq2SeqTrainer)

# load dataset
dataset = load_dataset("qanastek/ELRC-Medical-V2", "en-fr")
ds = dataset["train"].train_test_split(test_size=0.2, seed=42) if "train" in dataset else dataset.train_test_split(test_size=0.2, seed=42)

# load pretrained model and tokenizer
model_name = "Helsinki-NLP/opus-mt-en-fr"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

def tokenize_fn(examples):
    en_texts = [ex["en"] for ex in examples["translation"]]
    fr_texts = [ex["fr"] for ex in examples["translation"]]
    model_inputs = tokenizer(en_texts, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(fr_texts, max_length=128, truncation=True)
    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

tokenized_ds = ds.map(tokenize_fn, batched=True)# data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="./results",
    evaluation_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    weight_decay=0.01,
    save_total_limit=3,
    num_train_epochs=3,
    predict_with_generate=True,
    fp16=True,
    logging_dir="./logs"
)

# load bleu metric using evaluate
metric = evaluate.load("sacrebleu")

def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[lbl.strip()] for lbl in decoded_labels]
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {"bleu": result["score"]}

# setup trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_ds["train"],
    eval_dataset=tokenized_ds["test"],
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

# fine-tune the model
trainer.train()

# evaluate the model on the test set
eval_results = trainer.evaluate(tokenized_ds["test"])
print("evaluation metrics:", eval_results)

# get a few original examples from the test set (for english and reference french)
sample_original = ds["test"].select(range(5))

# get predictions from the fine-tuned (sft) model
ft_predictions = trainer.predict(sample_test)
ft_decoded_preds = tokenizer.batch_decode(ft_predictions.predictions, skip_special_tokens=True)


# load the base model and set up a trainer for it with an eval_dataset
base_model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
base_trainer = Seq2SeqTrainer(
    model=base_model,
    args=training_args,  # reuse same args
    eval_dataset=sample_test,  # add this line
    tokenizer=tokenizer,
    data_collator=data_collator
)
base_predictions = base_trainer.predict(sample_test)
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

