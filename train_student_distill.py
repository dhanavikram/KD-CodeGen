"""
Train student model using ONLY teacher probability distribution (pure KD).

Usage:
    python train_student_distill.py --data_dir data --output_dir checkpoints/student_kd
"""
import argparse, os
from datasets import load_from_disk
import torch
from transformers import (AutoTokenizer, AutoModelForSeq2SeqLM,
                          DataCollatorForSeq2Seq, Trainer, TrainingArguments)
from utils import set_seed, get_device

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data_dir", type=str, default="data")
    p.add_argument("--teacher_model", type=str, default="Salesforce/codet5-base")
    p.add_argument("--student_model", type=str, default="Salesforce/codet5-small")
    p.add_argument("--epochs", type=int, default=1)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--temperature", type=float, default=2.0)
    p.add_argument("--output_dir", type=str, default="checkpoints/student_kd")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()

def kd_loss(student_logits, teacher_logits, temperature):
    import torch.nn.functional as F
    t = temperature
    s_log_prob = F.log_softmax(student_logits / t, dim=-1)
    t_prob = F.softmax(teacher_logits / t, dim=-1)
    return F.kl_div(s_log_prob, t_prob, reduction="batchmean") * (t ** 2)

class KDTrainer(Trainer):
    def __init__(self, teacher, temperature=2.0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.teacher = teacher
        self.temperature = temperature

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels")  # remove but unused
        with torch.no_grad():
            teacher_out = self.teacher(**inputs)
        outputs = model(**inputs)
        loss = kd_loss(outputs.logits, teacher_out.logits, self.temperature)
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
    teacher.eval()
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
        bf16=False,
        fp16=False
    )

    trainer = KDTrainer(
        teacher=teacher,
        temperature=args.temperature,
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
