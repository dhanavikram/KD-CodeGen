#!/usr/bin/env python
"""
Aggregate one or more evaluation-JSON files and
 • print a Markdown metrics table
 • save a bar-chart as PNG/PDF

Usage:
    python visualize_results.py results/*.json --save_fig bleu.png
"""
import argparse, json, pandas as pd, matplotlib.pyplot as plt, textwrap, pathlib

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("json_files", nargs="+",
                   help="Metrics JSONs produced by evaluate.py")
    p.add_argument("--save_fig", type=str, default="bleu_comparison.png",
                   help="Filename for the bar-chart (extension decides format)")
    return p.parse_args()

def main():
    args = parse_args()

    rows = []
    for jf in args.json_files:
        with open(jf) as f:
            data = json.load(f)
        label = pathlib.Path(jf).stem        # e.g. student_mixed
        rows.append({"model": label,
                     "BLEU": data["student_BLEU"] if "student_BLEU" in data
                             else data["teacher_BLEU"]})
    df = pd.DataFrame(rows).set_index("model")
    df = df.sort_values("BLEU", ascending=False)

    # -------- table (Markdown) --------
    md = df.to_markdown(floatfmt=".2f")
    print("\n" + textwrap.indent(md, " " * 2) + "\n")

    # -------- chart --------
    df.plot(kind="bar", legend=False)
    plt.ylabel("BLEU ↑")
    plt.title("Teacher vs. Student BLEU")
    plt.tight_layout()
    plt.savefig(args.save_fig, dpi=200)
    print(f"Chart saved to {args.save_fig}")

if __name__ == "__main__":
    main()
