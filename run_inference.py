# run_inference.py
import json
import re
import csv

from pathlib import Path

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

    # Load the dataset from the input path
    data = [json.loads(line) for line in open(input_path)]

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
        system, user = build_prompt(item["question"], item.get("options"), variant)
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
    
    # Export results to CSV
    SAVE_EVAL = False  
    OUTPUT_PATH = "results/starter_results.csv"
    out_path = Path(OUTPUT_PATH)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(out_path, "w", newline="", encoding="utf-8") as f:
        
        if SAVE_EVAL:
            fieldnames = ["id", "is_mcq", "gold", "response", "correct"]
        else:
            fieldnames = ["id", "response"]
            
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        
        #Write the header row at the top of the CSV
        writer.writeheader()
        
        #Write the data rows
        for r in results:
            if SAVE_EVAL:
                record = {"id": r["id"], "is_mcq": r["is_mcq"], "gold": r["gold"],
                          "response": r["response"], "correct": r["correct"]}
            else:
                record = {"id": r["id"], "response": r["response"]}
                
            writer.writerow(record)
    
    print(f"Saved {len(results)} records to {out_path}")

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


