# run_inference.py

import json
import re
from pathlib import Path

import pandas as pd
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/your-base-model-or-final-model"
PROMPT_VARIANT = "multiple_answers"

MAX_TOKENS = 2048
TEMPERATURE = 0.0
TOP_P = 1.0
GPU_MEMORY_UTILIZATION = 0.90
MAX_MODEL_LEN = 4096

DEFAULT_INPUT_PATH = "data/private_test.json"
DEFAULT_OUTPUT_PATH = "outputs/submission.csv"

SYSTEM_PROMPT_MATH = (
    "Solve the math problem. Show only the necessary reasoning. "
    "Final answer rules:"
    "- Include EVERY requested answer in the final answer."
    "- If the problem has multiple parts or multiple blanks, put all answers in ONE final \\boxed{...}, separated by commas."
    "- Use exact form."
    "- No decimal approximations."
    "- Keep expressions symbolic."
    "- Use \\frac{}{}, powers, \\sqrt{}, \\pi, \\ln{}, \\arctan{} when appropriate."
    "- Only use decimals for numbers that are already decimals in the problem."
    "- End with the final answer in \\boxed{...}."
)

SYSTEM_PROMPT_MCQ = (
    "Solve the multiple-choice math problem. "
    "Output ONLY the correct choice letter inside \\boxed{}, e.g. \\boxed{C}."
)

'''
Run inference on the test dataset.
'''
def run_inference(input_path=DEFAULT_INPUT_PATH, output_path=DEFAULT_OUTPUT_PATH):

    load_dataset()
    prompt_construction()
    load_model()
    generate_predictions()
    score_responses()
    summarize_results()

    pass

'''
Load the test dataset from the specified input path.
'''
def load_dataset():
    pass

'''
Construct prompts for the model based on the test dataset.
'''
def prompt_construction():
    pass

'''
Load the pre-trained model for inference.
'''
def load_model():
    pass

'''
Generate predictions using the loaded model and constructed prompts.
'''
def generate_predictions():
    pass

'''
Score the generated responses against the ground truth answers.
'''
def score_responses():
    pass

'''
Summarize the results of the inference, including metrics and insights.
'''
def summarize_results():
    pass

'''
Save the results to the specified output path in the required format.
'''
def save_results():
    pass