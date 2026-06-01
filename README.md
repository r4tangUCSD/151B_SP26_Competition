# CSE 151B Competition — Starter Code

Ryan Tang,
Dylan Craver,
Michael Luo,
Linus Lee

## GPU & Inference Time

- **GPU**: NVIDIA A100 (80GB) via Google Colab Pro
- **Approximate inference time**: ~2 hours for 943 questions 

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

run_inference() takes in three arguments: an input path, and output path, and an optional N = number of examples. When N = None, it runs on the full dataset.

In run_inference.ipynb is the Jupyter notebook used to produce the final submission on Kaggle. Since it was ran in Colab, it mounts to Google Drive and uses Drive path names. To run it elsewhere, you may need to change cells with imports and file paths.
