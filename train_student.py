"""
Fine‑tune a student model on (docstring -> code) with optional knowledge distillation.

Usage:
    python train_student.py --data_dir data --output_dir checkpoints/student_mixed
"""
import argparse, os, math, time
from datasets import load_from_disk
import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from utils import set_seed, get_device

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data",
                   help="Directory containing tokenized train/validation splits")
    p.add_argument("--teacher_model", type=str, default="Salesforce/codet5-base")
    p.add_argument("--student_model", type=str, default="Salesforce/codet5-small")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--distill_alpha", type=float, default=0.5,
                   help="Coefficient for KD loss. 0 = no distillation, 1 = only distillation.")
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--output_dir", type=str, default="checkpoints/student_mixed")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def kd_loss(student_logits, teacher_logits, temperature):
    """KL divergence between teacher and student logits."""
    import torch.nn.functional as F
    t = temperature
    s_log_prob = F.log_softmax(student_logits / t, dim=-1)
    t_prob = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (t ** 2)

class DistilTrainer(Trainer):
    def __init__(self, teacher, temperature=2.0, alpha=0.5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = temperature
        self.alpha = alpha
        self.ce_loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")
        with torch.no_grad():
            teacher_out = self.teacher(**inputs)
        outputs = model(**inputs)
        student_logits = outputs.logits
        teacher_logits = teacher_out.logits

        loss_kd = kd_loss(student_logits, teacher_logits, self.temperature)
        loss_ce = self.ce_loss_fct(
            student_logits.view(-1, student_logits.size(-1)),
            labels.view(-1)
        )
        loss = self.alpha * loss_kd + (1 - self.alpha) * loss_ce
        if return_outputs:
            return loss, outputs
        return loss

def main():
    args = parse_args()
    set_seed(args.seed)
    device = get_device()

    tokenizer = AutoTokenizer.from_pretrained(args.student_model)
    student = AutoModelForSeq2SeqLM.from_pretrained(args.student_model).to(device)
    teacher = AutoModelForSeq2SeqLM.from_pretrained(args.teacher_model).to(device)
    teacher.eval()  # freeze teacher
    for p in teacher.parameters():
        p.requires_grad = False

    train_ds = load_from_disk(os.path.join(args.data_dir, "train.arrow"))
    val_ds = load_from_disk(os.path.join(args.data_dir, "validation.arrow"))

    data_collator = DataCollatorForSeq2Seq(tokenizer, model=student)
    args_tr = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        evaluation_strategy="epoch",
        save_strategy="epoch",
        logging_steps=50,
        learning_rate=5e-5,
        bf16=False,  # Apple silicon does not support bf16
        fp16=False
    )

    trainer = DistilTrainer(
        teacher=teacher,
        temperature=args.temperature,
        alpha=args.distill_alpha,
        model=student,
        args=args_tr,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        tokenizer=tokenizer,
        data_collator=data_collator,
    )
    trainer.train()
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

if __name__ == "__main__":
    main()
