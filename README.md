# Knowledge Distillation for Code Generation

This repository contains code to replicate the experiments
described in *Distilling Code Intelligence: A Knowledge Distillation Framework for Code Generation* 

## Folder structure
```
data_fetching.py          # Download & tokenize dataset
train_student.py          # Fine‑tune student with CE + KD loss
train_student_distill.py  # Train student using *only* teacher probabilities
evaluate.py               # Compute BLEU on the test set
utils.py                  # Common helpers (seed, device)
requirements.txt          # Locked versions for reproducibility
```
Output checkpoints are saved in `checkpoints/`, tokenized datasets in `data/`.

## 1️⃣ Setup (one‑time)
```bash
python3 -m venv venv
source venv/bin/activate
# Apple Silicon build of PyTorch comes with MPS‑acceleration
pip install -r requirements.txt
export PYTORCH_ENABLE_MPS_FALLBACK=1   # smoother experience on macOS
```

## 2️⃣ Data
```bash
python data_fetching.py --language python --sample_size 20000
```

*Change* `--sample_size` to `None` for the full dataset, but 20k keeps RAM
usage under 8 GB.

## 3️⃣ Training
**Mixed CE+KD**  
```bash
python train_student.py --data_dir data           --output_dir checkpoints/student_mixed           --epochs 3 --distill_alpha 0.5
```

**Pure KD**  
```bash
python train_student_distill.py --data_dir data           --output_dir checkpoints/student_kd           --epochs 3
```

## 4️⃣ Evaluation
```bash
python evaluate.py --data_dir data           --student_ckpt checkpoints/student_mixed
```

BLEU is reported for teacher and student; 

**TODO**: extend `evaluate.py` to add CodeBLEU, latency, and memory evaluations.

## Notes
* Dataset → **CodeSearchNet** 
* Teacher → `Salesforce/codet5-base` (≈220 M params).  
  Student → `Salesforce/codet5-small` (≈60 M).
* Hidden‑state alignment omitted for brevity; logits KD + CE proven effective.
* Metrics simplified to BLEU for fast evaluation; CodeBLEU can be added.
