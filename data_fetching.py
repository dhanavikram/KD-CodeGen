#!/usr/bin/env python
"""
Download and preprocess CodeSearchNet for code generation distillation.

Example:
    python data_fetching.py --language python --sample_size 20000
"""
import argparse, os, random, multiprocessing as mp
from datasets import load_dataset
from transformers import AutoTokenizer
from utils import set_seed

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--language", type=str, default="python",
                   help="Programming language subset to use")
    p.add_argument("--teacher_model", type=str, default="Salesforce/codet5-base")
    p.add_argument("--sample_size", type=int, default=None,
                   help="If set, subsample this many examples from each split for quick experiments")
    p.add_argument("--output_dir", type=str, default="data", help="Directory to save tokenized dataset")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def main():
    args = parse_args()
    set_seed(args.seed)
    os.makedirs(args.output_dir, exist_ok=True)

    splits = ["train", "validation", "test"]
    ds = {}
    print("Loading CodeSearchNet...")
    for split in splits:
        subset = load_dataset("code_search_net", args.language, split=split)
        if args.sample_size:
            subset = subset.shuffle(seed=args.seed).select(range(args.sample_size))
        ds[split] = subset
    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)

    def tokenize(ex):
        source = ex["func_documentation_string"] or ""
        target = ex["func_code_string"] or ""
        model_inputs = tokenizer(source, truncation=True, padding="max_length", max_length=256)
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(target, truncation=True, padding="max_length", max_length=256)
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs

    for split in splits:
        proc = min(mp.cpu_count(), 4)
        toks = ds[split].map(tokenize, batched=True, num_proc=proc,
                             remove_columns=ds[split].column_names,
                             desc=f"Tokenizing {split}")
        path = os.path.join(args.output_dir, f"{split}.arrow")
        toks.save_to_disk(path)
        print(f"Saved {split} to {path}")

if __name__ == "__main__":
    main()
