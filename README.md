# CSE 151B Competition — Starter Code

Ryan Tang
Dylan Craver
Michael Luo
Linus Lee

Open **`starter_code_cse151b_comp.ipynb`** to get started.

The notebook covers environment setup, inference with Qwen3-4B-Thinking (INT8), and scoring against the public dataset.

## Contents

| File | Description |
|---|---|
| `starter_code_cse151b_comp.ipynb` | Main entry point |
| `judger.py` | Response scoring logic |
| `utils.py` | Utilities used by `judger.py` |
| `data/public.jsonl` | Public dataset with ground-truth answers |
| `results/` | Output JSONL files written at runtime |

## GPU & Inference Time

- **GPU**: NVIDIA A100 (80GB) via Google Colab Pro
- **Approximate inference time**: ~2 hours for 1,126 questions 

## Model Weights

No local download required. The model is hosted on HuggingFace and loaded automatically.

Update `MODEL_ID` in `run_inference.py` to point to your model:

```python
MODEL_ID = "Qwen/Qwen3-4B-Thinking-2507" for baseline
```
## How to Run Inference

```bash
python run_inference.py
```

Or call directly:

```python
from run_inference import run_inference
run_inference(input_path="data/private_test.json", output_path="outputs/submission.csv")
```