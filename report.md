# Enhancing Machine Translation with Fine-Tuning
**A Comparative Study on English-to-French Translation Using Pretrained and Fine-Tuned Models**

## Introduction
- **Objective:** Improve machine translation by fine-tuning a pretrained model on a medical translation dataset.
- **Dataset:** [ELRC-Medical-V2 (English-French)](https://huggingface.co/datasets/qanastek/ELRC-Medical-V2)
- **Model:** `Helsinki-NLP/opus-mt-en-fr` (English-to-French Translation)
- **Evaluation Metric:** BLEU Score (via `sacrebleu`)

## Methodology
### 1️Dataset Preparation
- Loaded the **ELRC-Medical-V2** dataset.
- Split into **80% training / 20% test**.
- Tokenized English-French sentence pairs.

### 2️Pretrained Model Setup
- Loaded **Helsinki-NLP/opus-mt-en-fr**.
- Tokenized inputs and outputs.
- Used `DataCollatorForSeq2Seq` to batch process data.

### 3️Fine-Tuning
- Used **Seq2SeqTrainer** with:
  - **Learning rate:** `2e-5`
  - **Batch size:** `16`
  - **Epochs:** `3`
  - **Weight decay:** `0.01`
- Fine-tuned on the dataset.

### 4️Evaluation
- Computed BLEU scores for:
  - **Pretrained model**
  - **Fine-tuned model**
- Compared translations of sample sentences.

## Results
### Training Performance
| Epoch | Training Loss | Validation Loss | BLEU Score |
|-------|--------------|----------------|------------|
| 1     | 0.636       | 0.551          | 57.84      |
| 2     | 0.538       | 0.539          | 58.44      |
| 3     | 0.496       | 0.537          | 58.53      |

- BLEU score **improved slightly** after fine-tuning.

### Example Comparisons
| **English Input** | **Pretrained Model** | **Fine-Tuned Model** | **Reference Translation** |
|------------------|---------------------|---------------------|-------------------------|
| "To encourage governments to adopt strategies..." | "Avant de tirer ses conclusions, il s'est engagé..." | "Avant de tirer ses conclusions, elle s'est engagée..." | "Avant de tirer ses conclusions, elle a dialogué..." |
| "Alert notification and public health risk assessment" | "Avis d'alerte et évaluation des risques..." | "Notification d'alerte et évaluation des risques..." | "Notification d'alertes et évaluation des risques..." |

- **Fine-tuned model produces more natural and accurate translations.**

## Conclusions
- Fine-tuning **improves domain-specific translation**.
- BLEU score improvements were **marginal**, but translations **show qualitative enhancement**.
- **Future Work:**
  - Train on a **larger dataset**.
  - Use **more epochs** or **adjust hyperparameters**.
  - Experiment with **larger transformer models**.

## Cool Stats
- **Total FLOPs:** `6.02e+14` FLOPs (**602 trillion computations**)
- **Training Time:** `806.38` seconds (~13.44 minutes)
- **FLOPs per Second:** `7.46e+11` FLOPs/sec (**746 billion computations per second**)
- **FLOPs per Training Step:** `3.05e+11` FLOPs/step (**305 billion computations per step**)
- **Training Steps:** `1,974`
- **Samples Processed per Second:** `39.13`
- **Steps Processed per Second:** `2.45`