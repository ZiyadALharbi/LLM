# Building an LLM From Scratch

This repository is a hands-on learning project for building GPT-style language
models from the ground up with PyTorch. The notebooks keep the chapter-by-chapter
experiments, and `complete_version/` contains the reusable Python implementation.

## Project Layout

```text
.
├── complete_version/          # Clean Python package version
├── notebooks/                 # Learning notebooks, numbered by topic order
├── data/                      # Small local datasets used by notebooks/scripts
├── assets/pretraining_images/ # Images saved from notebook work
├── gpt.py                     # Compatibility exports for older notebook imports
├── previous_chapters.py       # Compatibility exports for chapter 7 imports
├── coding_attention_mechanisms.py
├── requirements.txt
└── LICENSE
```

## Notebooks

- `notebooks/01_pytorch_basics.ipynb`
- `notebooks/02_working_with_text_data.ipynb`
- `notebooks/03_attention_mechanisms.ipynb`
- `notebooks/04_gpt_model.ipynb`
- `notebooks/05_pretraining.ipynb`
- `notebooks/07_instruction_finetuning.ipynb`

The notebooks are still useful for studying each concept in order. For reusable
code, prefer importing from `complete_version`.

## Complete Python Version

The complete Python version is split by responsibility:

- `complete_version/dataset.py`: tokenized sliding-window dataset and dataloader
- `complete_version/attention.py`: multi-head causal self-attention
- `complete_version/model.py`: LayerNorm, GELU, feed-forward, transformer block, GPT model
- `complete_version/generation.py`: greedy, temperature, and top-k generation helpers
- `complete_version/training.py`: loss, evaluation, and simple training loop
- `complete_version/instruction_finetuning.py`: instruction dataset and collate function
- `complete_version/gpt2_weights.py`: GPT-2 checkpoint loading helpers
- `complete_version/run_demo.py`: quick untrained model demo

Compatibility modules are kept at the repo root so older notebook cells such as
`from gpt import GPTModel` and `from previous_chapters import generate` continue
to work.

## Setup

Use Python 3.10 or newer.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Some instruction fine-tuning notebook cells load pretrained GPT-2 weights using
`gpt_download`. If that helper is not available locally, install the companion
package used by the book notebooks:

```bash
python -m pip install llms-from-scratch
```

If you run the GPT-2 weight-loading sections, TensorFlow may also be needed
because the original GPT-2 checkpoints are converted from TensorFlow format:

```bash
python -m pip install tensorflow
```

## Running The Python Version

Run the standalone demo from the repository root:

```bash
python -m complete_version.run_demo
```

This constructs an untrained GPT-style model, tokenizes a short prompt, and runs
greedy generation. The output will not be meaningful until the model has been
trained; the script mainly verifies that the architecture and forward pass work.

The old entry point still works:

```bash
python pretraining/gpt.py
```

## Data

Small local datasets are stored in `data/`:

- `data/the-verdict.txt`: sample text for tokenization and pretraining experiments
- `data/instruction-data.json`: instruction-response examples for supervised instruction tuning

Generated checkpoints and downloaded GPT-2 weights are ignored by git.

## Notes

- This is an educational project, not a production LLM training framework.
- The notebooks intentionally build many concepts from scratch for learning clarity.
- Training quality depends heavily on model size, data size, training time, and hardware.

## License

This project is licensed under the MIT License. See `LICENSE` for details.
