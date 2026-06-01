# run_inference.py
import json
import re
import csv

from pathlib import Path
from typing import Optional
from transformers import AutoTokenizer
from vllm import LLM, SamplingParams

MODEL_ID = "Qwen/your-base-model-or-final-model"

MAX_TOKENS = 32768

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
def run_inference(input_path=DEFAULT_INPUT_PATH, output_path=DEFAULT_OUTPUT_PATH, N=None):
    """
    Run inference on the test dataset.

    Args:
        input_path: Path to the JSONL test file.
        output_path: Path where the submission CSV should be written.
        N: Optional number of examples to run for debugging. If None, run the full dataset.
    """

    # Load the dataset from the input path
    with open(input_path, "r", encoding="utf-8") as f:
        data = [json.loads(line) for line in f]

    if N is not None:
        if N <= 0:
            raise ValueError("N must be a positive integer or None.")
        data = data[:N]
        print(f"Debug mode: running first {len(data)} examples.")
    else:
        print(f"Full mode: running all {len(data)} examples.")

    # Load the pre-trained model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
    tokenizer.pad_token = tokenizer.eos_token

    llm = LLM(
        model=MODEL_ID,
        quantization="bitsandbytes",
        load_format="bitsandbytes",
        enable_prefix_caching=False,
        gpu_memory_utilization=0.50,
        max_model_len=16384,
        trust_remote_code=True,
        max_num_seqs=256,
        max_num_batched_tokens=32768,
    )

    sampling_params = SamplingParams(
        max_tokens=MAX_TOKENS,
        temperature=0.6,
        top_p=0.95,
        top_k=20,
        min_p=0.0,
        presence_penalty=0.0,
        repetition_penalty=1.0,
    )

    print("Model loaded.")

    # Build prompts for each question in the dataset
    prompts = []
    for item in data:
        system, user = build_prompt(item["question"], item.get("options"))
        prompt_text = tokenizer.apply_chat_template(
            [{"role": "system", "content": system},
            {"role": "user",   "content": user}],
            tokenize=False,
            add_generation_prompt=True,
        )
        prompts.append(prompt_text)

    # Generate predictions with VLLM
    responses = []

    print(f"Generating responses for {len(prompts)} questions")

    outputs = llm.generate(prompts, sampling_params=sampling_params)

    responses = [out.outputs[0].text.strip() for out in outputs]

    # Export results to CSV
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["id", "response"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)

        writer.writeheader()

        for item, response in zip(data, responses):
            record = {
                "id": item["id"],
                "response": response,
            }
            writer.writerow(record)

    print(f"Saved {len(responses)} records to {output_path}")
    return output_path

'''
Build the system and user prompts based on the question type
'''
def build_prompt(question: str, options: Optional[list]) -> tuple[str, str]:
    """Return (system_prompt, user_prompt) for a question."""
    if options:
        labels    = [chr(65 + i) for i in range(len(options))]
        opts_text = "\n".join(f"{lbl}. {opt.strip()}" for lbl, opt in zip(labels, options))
        return SYSTEM_PROMPT_MCQ, f"{question}\n\nOptions:\n{opts_text}"
    return SYSTEM_PROMPT_MATH, question


