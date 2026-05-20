# DA6401 - Assignment 3: Implementing the Transformer for Machine Translation

## Overview

In this assignment, you will implement the landmark architecture from the paper "Attention Is All You Need" from scratch using PyTorch. The goal is to develop a Neural Machine Translation (NMT) system capable of translating text from German to English using the Multi30k dataset.

## Project Structure

```text
assignment3/
├── requirements.txt
├── README.md
├── model.py           # Core Transformer architecture (Encoders, Decoders, Multi-Head Attention)
├── utils.py           # Label Smoothing, Noam Scheduler, Masking Utilities
├── dataset.py         # Multi30k dataset loading and spacy tokenization
├── train.py           # Training loops and Greedy Decoding inference
```
MY WANDB REPORT LINK : https://wandb.ai/ee23b052-iitm-india/DA_DL_Ass3/reports/DA6401-Assignment-3-Transformer-Analysis--VmlldzoxNjk0NzUxOQ

## Files

- model.py — Transformer model implementation
- dataset.py — Multi30k dataset processing
- lr_scheduler.py — Noam learning-rate scheduler
- train.py — Training pipeline
- PJ_2.1.py — Noam Scheduler experiment
- PJ_2.2.py — Scaling factor experiment
- PJ_2.3.py — Attention visualization experiment
- PJ_2.4.py — Label smoothing experiment
- generate_all_visualizations.py — Final visualization generator