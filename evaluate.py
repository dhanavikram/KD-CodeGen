#!/usr/bin/env python
"""
Extended evaluation:
  • BLEU
  • average / p95 generation latency (ms)
  • peak Δ-RSS memory used during generation (MB)
  • parameter count (M) and approximate disk footprint (MB)

Example
-------
python evaluate.py \
        --data_dir data \
        --student_ckpt checkpoints/student_mixed \
        --outfile results/resources_student.json
"""
import argparse, os, time, json, statistics, psutil, torch
from datasets import load_from_disk
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sacrebleu import corpus_bleu
from tqdm.auto import tqdm


# ---------- helpers ----------------------------------------------------------
def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")                  # Apple-Silicon GPU
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def param_stats(model):
    n_params = sum(p.numel() for p in model.parameters())
    approx_mb = n_params * 2 / 1024**2              # 2 bytes ≈ bf16
    return n_params / 1e6, approx_mb                # (M), (MB)


def generate_one(model, tok, prompt, beam, dev):
    ids = tok(prompt, return_tensors="pt",
              truncation=True, padding=True).to(dev)
    with torch.no_grad():
        out = model.generate(**ids,
                             max_length=256,
                             num_beams=beam)
    return tok.decode(out[0], skip_special_tokens=True)


def eval_model(label, model, tok, prompts, refs, beam, dev):
    proc       = psutil.Process()
    base_rss   = proc.memory_info().rss
    peak_rss   = base_rss
    latencies  = []
    predictions = []

    for p in tqdm(prompts, desc=f"{label:>8}"):
        t0 = time.perf_counter()
        predictions.append(generate_one(model, tok, p, beam, dev))
        latencies.append((time.perf_counter() - t0) * 1_000)    # ms
        peak_rss = max(peak_rss, proc.memory_info().rss)

    return {
        "BLEU"              : corpus_bleu(predictions, [refs]).score,
        "avg_latency_ms"    : statistics.mean(latencies),
        "p95_latency_ms"    : statistics.quantiles(latencies, n=20)[-1],
        "peak_delta_mem_MB" : (peak_rss - base_rss) / 1024**2
    }


# ---------- main -------------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir",      default="data")
    p.add_argument("--teacher_model", default="Salesforce/codet5-base")
    p.add_argument("--student_ckpt",  required=True)
    p.add_argument("--beam_size",     type=int, default=5)
    p.add_argument("--max_eval",      type=int, default=1000,
                   help="Evaluate on this many examples (use None for all)")
    p.add_argument("--outfile",       default="results/resources.json")
    return p.parse_args()


def main():
    args   = parse_args()
    device = get_device()
    os.makedirs(os.path.dirname(args.outfile), exist_ok=True)

    tok     = AutoTokenizer.from_pretrained(args.teacher_model)
    test_ds = load_from_disk(os.path.join(args.data_dir, "test.arrow"))
    if args.max_eval:
        test_ds = test_ds.select(range(args.max_eval))

    prompts = [tok.decode(r["input_ids"], skip_special_tokens=True) for r in test_ds]
    refs    = [tok.decode(r["labels"],    skip_special_tokens=True) for r in test_ds]

    results = {}
    for label, path in {"teacher": args.teacher_model,
                        "student": args.student_ckpt}.items():
        print(f"\nLoading **{label}** …")
        model = AutoModelForSeq2SeqLM.from_pretrained(path).to(device)
        model.eval()

        # core metrics
        metrics = eval_model(label, model, tok, prompts, refs,
                             args.beam_size, device)

        # static model stats
        nM, mb = param_stats(model)
        metrics.update({"params_M": nM, "approx_size_MB": mb})
        results[label] = metrics

        # release GPU / MPS RAM
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    with open(args.outfile, "w") as f:
        json.dump(results, f, indent=2)
    print("\n" + json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
