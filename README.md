# Building an LLM From Scratch

This repository is a hands-on learning project for building GPT-style language models from the ground up with PyTorch. It starts with text tokenization and dataset preparation, then builds attention mechanisms, a GPT model, pretraining loops, and two fine-tuning workflows:

- SMS spam classification
- Instruction following

The code is organized as notebooks for experimentation, plus a reusable `pretraining/gpt.py` module that collects the core model components.

## What Is Implemented

### Text Data Preparation

`working_with_text_data/working_with_text_data.ipynb` covers the early data pipeline:

- basic regex tokenization
- simple tokenizer classes
- special tokens
- GPT-2 byte pair encoding with `tiktoken`
- sliding-window dataset sampling
- token embeddings
- positional embeddings

### Attention Mechanisms

`coding_attention_mechanisms/attention.ipynb` builds attention step by step:

- self-attention
- causal masking
- dropout in attention
- multi-head attention wrappers
- a compact multi-head attention module

### GPT Model Components

`GPT_model/GPT_model.ipynb` develops the transformer model pieces:

- token and positional embeddings
- layer normalization
- GELU activation
- feed-forward blocks
- shortcut connections
- transformer blocks
- a GPT-style model skeleton

The reusable implementation lives in `pretraining/gpt.py` and includes:

- `GPTDatasetV1`
- `create_dataloader_v1`
- `MultiHeadAttention`
- `LayerNorm`
- `GELU`
- `FeedForward`
- `TransformerBlock`
- `GPTModel`
- `generate_text_simple`

### Pretraining

`pretraining/pretraining_on_unlabeled_data.ipynb` trains a small GPT-style model on `the-verdict.txt` and includes:

- training and validation loss calculation
- a simple training loop
- text generation during training
- temperature sampling
- top-k sampling
- model and optimizer checkpoint saving

### Fine-Tuning for Text Classification

`Finetuning for Text Classification/finetuning_for_text_classification.ipynb` fine-tunes a GPT-style model as a binary spam classifier using the SMS Spam Collection dataset.

The folder includes prepared splits:

- `train.csv`
- `validation.csv`
- `test.csv`

Each CSV has:

- `Label`: `0` for ham, `1` for spam
- `Text`: SMS message text

### Fine-Tuning for Instruction Following

`finetuning_for_instruction_following/ch7.ipynb` fine-tunes a GPT-style model on instruction-response examples from `instruction-data.json`.

The instruction dataset contains entries with:

- `instruction`
- `input`
- `output`

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install torch tiktoken pandas matplotlib requests tqdm jupyter
```

Some notebooks load pretrained GPT-2 weights using helper modules named `gpt_download` and `previous_chapters`. If those files are not available locally, install the companion package used by the notebooks:

```bash
python -m pip install llms-from-scratch
```

If you run the GPT-2 weight-loading sections, you may also need TensorFlow because the original GPT-2 checkpoints are converted from TensorFlow format:

```bash
python -m pip install tensorflow
```

## Running the Reusable GPT Module

You can run the standalone GPT module directly:

```bash
python pretraining/gpt.py
```

This constructs an untrained GPT-style model, tokenizes a short prompt, and performs simple greedy generation. The output will not be meaningful unless the model has been trained; the script is mainly useful for verifying that the architecture and forward pass work.

## Data

This repository includes small local datasets used by the notebooks:

- `the-verdict.txt`: sample text used for tokenization and pretraining experiments
- SMS Spam Collection files: raw and prepared data for binary text classification
- `instruction-data.json`: instruction-response examples for supervised instruction tuning

The SMS dataset includes its original readme at:

```text
Finetuning for Text Classification/sms_spam_collection/readme
```

## Notes

- This is an educational project, not a production LLM training framework.
- The notebooks intentionally build many concepts from scratch for learning clarity.
- Training quality depends heavily on model size, data size, training time, and hardware.
- The fine-tuning notebooks may require downloading pretrained GPT-2 weights before the later cells can run.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
