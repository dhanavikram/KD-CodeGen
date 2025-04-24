"""
Evaluate teacher and student models on the CodeSearchNet test split.

Example:
    python evaluate.py --data_dir data --student_ckpt checkpoints/student_mixed --metrics_file results/student_mixed.json
"""
import argparse, os, time, json
import torch, psutil
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from utils import set_seed, get_device
from sacrebleu import corpus_bleu
from tqdm.auto import tqdm

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--teacher_model", type=str, default="Salesforce/codet5-base")
    p.add_argument("--student_ckpt", type=str, required=True,
                   help="Path to fine‑tuned student checkpoint")
    p.add_argument("--beam_size", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max_eval", type=int, default=1000,
                   help="How many examples to evaluate (None = all)")
    p.add_argument("--metrics_file", type=str, default=None,
                   help="If set, dump scores to this JSON file")
    return p.parse_args()

def generate(model, tokenizer, inputs, beam_size=5, device="cpu"):
    model.eval()
    with torch.no_grad():
        ids = tokenizer(inputs, return_tensors="pt", truncation=True,
                        padding=True).to(device)
        gen_ids = model.generate(**ids, max_length=256, num_beams=beam_size)
    return tokenizer.batch_decode(gen_ids, skip_special_tokens=True)

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.teacher_model)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model).to(device)
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student_ckpt).to(device)

    test_ds = load_from_disk(os.path.join(args.data_dir, "test.arrow"))
    if args.max_eval:
        test_ds = test_ds.select(range(args.max_eval))

    references = []
    teacher_preds = []
    student_preds = []

    for batch in tqdm(test_ds, desc="Evaluating"):
        doc = tokenizer.decode(batch["input_ids"], skip_special_tokens=True)
        ref = tokenizer.decode(batch["labels"], skip_special_tokens=True)
        references.append(ref)

        teacher_pred = generate(teacher, tokenizer, doc,
                                beam_size=args.beam_size, device=device)[0]
        student_pred = generate(student, tokenizer, doc,
                                beam_size=args.beam_size, device=device)[0]
        teacher_preds.append(teacher_pred)
        student_preds.append(student_pred)

    teacher_bleu = corpus_bleu(teacher_preds, [references]).score
    student_bleu = corpus_bleu(student_preds, [references]).score

    results = {
        "teacher_BLEU": teacher_bleu,
        "student_BLEU": student_bleu,
        "num_eval": len(references)
    }
    print(json.dumps(results, indent=2))
    
    # optional persistence for later plotting
    if args.metrics_file:
        os.makedirs(os.path.dirname(args.metrics_file), exist_ok=True)
        with open(args.metrics_file, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics to {args.metrics_file}")

if __name__ == "__main__":
    main()
