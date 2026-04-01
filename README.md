# IndicTrans2 LoRA Fine-Tuning (150K Dataset)

## Overview

This project fine-tunes the IndicTrans2 translation model using LoRA (Low-Rank Adaptation) on a parallel dataset of approximately 150K examples.

## What This Project Does

When you run the notebook, it will:

1. Load the pretrained IndicTrans2 model
2. Apply LoRA adapters for parameter-efficient fine-tuning
3. Load and clean the training dataset
4. Train the model on the source-target pairs
5. Evaluate the model using:
   - BLEU score
   - Semantic similarity
   - Generation throughput
6. Save the fine-tuned artifacts for later use

## What You Need Before Running

### Hardware
- Minimum: 1 GPU
- Recommended for this notebook: NVIDIA T4 or better
- More GPU memory will reduce training time

### Software
- Python 3.10+
- PyTorch
- Transformers
- Datasets
- PEFT
- SacreBLEU
- SentenceTransformers

### Install Dependencies

```bash
pip install torch transformers datasets peft accelerate sacrebleu sentence-transformers
```

## How to Run It

### 1. Open the notebook
Run the notebook in:
- Google Colab, or
- A local Jupyter environment with GPU support

### 2. Configure the language and column settings
Set the source language, target language, and the dataset column names.

Example:

```python
SRC_LANG = "eng_Latn"
TGT_LANG = "hin_Deva"

SRC_COL = "english"
TGT_COL = "hindi"

MAX_LEN = 128
```

### 3. Run training
Execute the notebook cells in order.

The notebook will:
- Load the base model
- Add LoRA adapters
- Train on the dataset
- Save the trained adapter weights

### 4. Run evaluation
After training, the notebook evaluates the model on a held-out test set.

It reports:
- BLEU score
- Semantic similarity
- Words per minute (throughput)

## What Happens During the Process

### Training
During training, only the LoRA layers are updated. This makes the process much cheaper and faster than full fine-tuning.

### Evaluation
The model is tested on unseen examples to measure:
- How closely the translation matches the reference
- Whether the meaning is preserved
- How fast the model generates output

### Output
At the end, you should expect:
- Fine-tuned LoRA adapter weights
- Evaluation scores
- Sample translations
- A model ready for inference or merging

## Expected Results

### BLEU Score
A higher BLEU score means better overlap with the reference translation.

Typical interpretation:
- 36+ = excellent
- 32–36 = strong
- 28–32 = acceptable
- Below 28 = needs improvement

### Semantic Similarity
This measures meaning preservation rather than exact wording.

A higher score means the output is closer in meaning to the reference translation.

### Words Per Minute
This measures generation speed. It is useful when the model is intended for production or real-time use.


## Summary

This project provides a practical pipeline to fine-tune IndicTrans2 on custom bilingual data with minimal compute cost and measurable evaluation output.
