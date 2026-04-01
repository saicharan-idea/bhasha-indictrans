# Bhaasha - Legal Document Translation to Indic Languages

## Overview
This repository contains a suite of notebooks and pipelines dedicated to benchmarking and fine-tuning state-of-the-art Neural Machine Translation (NMT) models for the complex task of translating English legal documents into Indic languages (with a prominent focus on Marathi).

## Supported Architectures
This repository contains complete pipelines for training, evaluating, and running inference on several architectures:
*   **NLLB-200 (No Language Left Behind)**: Memory-optimized fine-tuning of the `facebook/nllb-200-distilled-600M` model.
*   **IndicTrans2**: Inference and Zero-Shot baseline benchmarking using `ai4bharat/indictrans2-en-indic-1B`.
*   **mBART**: Specialized training and evaluation paths for mBART large.

## Repository Structure

*   `nllb-finetunning.ipynb`
    The core training loop for fine-tuning the 600M NLLB model entirely on consumer/free-tier GPUs (e.g., Kaggle T4, Google Colab 15GB). This notebook implements extreme memory optimizations (8-bit quantization, gradient checkpointing, mixed precision) to circumvent CUDA Out-Of-Memory errors while maintaining identical modeling accuracy.
*   `mbart_finetune_training.ipynb`
    Historical or specialized pipeline dedicated to the end-to-end fine-tuning and state management of mBART architectures.
*   `benchmark_translation.ipynb`
    Zero-shot and baseline generation scripts. Optimized for mass-translation generation combining dynamic VRAM caching fixes and tensor batching.
*   `eval_mbart.ipynb` & `eval_bhaasha.ipynb`
    Rigorous evaluation scripts computing `sacrebleu` metrics. Automatically compares Fine-Tuned BLEU scores against the Base Model BLEU scores, producing detailed visual predictions linking Source, Reference, and Predicted data.

## Key Technical Features

### Extreme Memory Optimizations
Pipelines are hardcoded to utilize `bitsandbytes` 8-bit AdamW (`optim="adamw_8bit"`), combined with `gradient_checkpointing=True` and Mixed Precision FP16 (`fp16=True`). By coupling drastically reduced real batch sizes (e.g., `4`) with expanded gradient accumulation steps (e.g., `8`), the code simulates massive 32-batch training steps entirely under 15GB of VRAM limits.

### Kaggle & Colab File Safety
Kaggle and Colab filesystems frequently desync or wipe `/kaggle/working/` directories upon kernel restarts or GPU limits. The pipelines avoid this by natively implementing a `.zip` state-save mechanism via `shutil.make_archive` at the split-second training ends, ensuring checkpoints are permanently accessible.

### In-Memory Evaluation
Evaluation pipelines are designed to load their models directly from PyTorch's `Seq2SeqTrainer` object memory reference (`ft_model = model`) rather than risking `HFValidationError` bugs during disk I/O reads.

## Basic Setup

### Core Dependencies
Ensure the compute instance is instantiated with the following dependencies to avoid mathematical un-scaler bounds errors:
```bash
pip install torch transformers datasets accelerate sacrebleu bitsandbytes
```

### Datasets
Requires highly clean parallel translation data provided as CSVs (e.g., `legal_train.csv`, `legal_test.csv`, `legal_val.csv`) containing source and benchmark target columns. The ingestion pipeline natively uses `os.walk` path discovery for flexible directory mounts on Kaggle.
