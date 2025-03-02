# Enhancing Machine Translation with Fine-Tuning
**A Comparative Study on English-to-French Translation Using Pretrained and Fine-Tuned Models**

## Introduction
- **Objective:** Improve machine translation by fine-tuning a pretrained model on a medical translation dataset.
- **Dataset:** [ELRC-Medical-V2 (English-French)](https://huggingface.co/datasets/qanastek/ELRC-Medical-V2)
- **Model:** `Helsinki-NLP/opus-mt-en-fr` (English-to-French Translation)
- **Evaluation Metric:** BLEU Score (via `sacrebleu`)

## Methodology
### Dataset Preparation
- Loaded the **ELRC-Medical-V2** dataset.
- Split into **80% training / 20% test**.
- Tokenized English-French sentence pairs.

### Pretrained Model Setup
- Loaded **Helsinki-NLP/opus-mt-en-fr**.
- Tokenized inputs and outputs.
- Used `DataCollatorForSeq2Seq` to batch process data.

### 3Fine-Tuning
- Used **Seq2SeqTrainer** with:
  - **Learning rate:** `2e-5`
  - **Batch size:** `16`
  - **Epochs:** `3`
  - **Weight decay:** `0.01`
- Fine-tuned on the dataset.

### Evaluation
- Computed BLEU scores for:
  - **Pretrained model**
  - **Fine-tuned model**
- Compared translations of sample sentences.

## Results
| Epoch | Training Loss | Validation Loss | BLEU     | METEOR   | BERTScore F1 |
|:-----:|:------------:|:---------------:|:--------:|:--------:|:------------:|
|   0   | N/A          | N/A             | 32.999   | 0.527    | 0.824        |
|   1   | 0.635400     | 0.549073        | 36.972185| 0.638258 | 0.864561     |
|   2   | 0.531700     | 0.535434        | 40.855252| 0.641706 | 0.866834     |
|   3   | 0.475200     | 0.545693        | 45.469537| 0.668619 | 0.878935     |
|   4   | 0.385800     | 0.535864        | 49.738398| 0.651694 | 0.870610     |


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
