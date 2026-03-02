"""
Train in-house NER model for symptom extraction.
Uses DistilBERT fine-tuned for token classification (B-SYMPTOM, I-SYMPTOM, O).

Usage:
  1. python scripts/prepare_ner_data.py   # Generate data first
  2. python scripts/train_ner.py          # Train model
"""

import sys
from pathlib import Path

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Check deps
try:
    import torch
    from transformers import (
        AutoTokenizer,
        AutoModelForTokenClassification,
        TrainingArguments,
        Trainer,
        DataCollatorForTokenClassification,
    )
    from datasets import Dataset
    import evaluate
except ImportError as e:
    print("Install required packages: pip install transformers torch datasets evaluate seqeval")
    raise SystemExit(1) from e


DATA_DIR = project_root / "data" / "ner"
MODEL_DIR = project_root / "models" / "ner_symptom"
LABELS = ["O", "B-SYMPTOM", "I-SYMPTOM"]
LABEL2ID = {l: i for i, l in enumerate(LABELS)}
ID2LABEL = {i: l for i, l in enumerate(LABELS)}


def load_conll(path: Path) -> list[dict]:
    """Load CoNLL file into list of {tokens, ner_tags}."""
    examples = []
    current_tokens = []
    current_labels = []

    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                if current_tokens:
                    tag_ids = [LABEL2ID.get(l, 0) for l in current_labels]
                    examples.append({"tokens": current_tokens, "ner_tags": tag_ids})
                    current_tokens = []
                    current_labels = []
                continue
            parts = line.split("\t")
            if len(parts) >= 2:
                current_tokens.append(parts[0])
                current_labels.append(parts[1])
            else:
                current_tokens.append(parts[0])
                current_labels.append("O")

    if current_tokens:
        tag_ids = [LABEL2ID.get(l, 0) for l in current_labels]
        examples.append({"tokens": current_tokens, "ner_tags": tag_ids})

    return examples


def tokenize_and_align_labels(examples, tokenizer):
    """Align tokenizer subwords with word-level labels."""
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        max_length=128,
        padding="max_length",
        return_tensors=None,
    )

    labels = []
    for i, label in enumerate(examples["ner_tags"]):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            if word_idx is None:
                label_ids.append(-100)
            elif word_idx != previous_word_idx:
                label_ids.append(label[word_idx])
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx
        labels.append(label_ids)

    tokenized_inputs["labels"] = labels
    return tokenized_inputs


def compute_metrics(p, label_list):
    """Seqeval metrics for NER."""
    predictions, labels = p
    predictions = predictions.argmax(axis=2)

    true_preds = []
    true_labels = []
    for pred, label in zip(predictions, labels):
        true_pred = [label_list[p] for (p, l) in zip(pred, label) if l != -100]
        true_label = [label_list[l] for (p, l) in zip(pred, label) if l != -100]
        true_preds.append(true_pred)
        true_labels.append(true_label)

    seqeval = evaluate.load("seqeval")
    results = seqeval.compute(predictions=true_preds, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def main():
    print("Loading data...")
    train_data = load_conll(DATA_DIR / "train.txt")
    val_data = load_conll(DATA_DIR / "val.txt")

    if not train_data:
        print("No training data. Run: python scripts/prepare_ner_data.py")
        sys.exit(1)

    train_ds = Dataset.from_list(train_data)
    val_ds = Dataset.from_list(val_data)

    print(f"Train: {len(train_ds)} | Val: {len(val_ds)}")

    model_name = "distilbert/distilbert-base-uncased"
    print(f"Loading tokenizer: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    def tokenize_fn(examples):
        return tokenize_and_align_labels(examples, tokenizer)

    print("Tokenizing...")
    train_tokenized = train_ds.map(tokenize_fn, batched=True, remove_columns=train_ds.column_names)
    val_tokenized = val_ds.map(tokenize_fn, batched=True, remove_columns=val_ds.column_names)

    data_collator = DataCollatorForTokenClassification(tokenizer=tokenizer)

    print(f"Loading model: {model_name}")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    training_args = TrainingArguments(
        output_dir=str(MODEL_DIR),
        num_train_epochs=2,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        learning_rate=2e-5,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        logging_steps=50,
        save_total_limit=2,
    )

    def compute_fn(p):
        return compute_metrics(p, LABELS)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_tokenized,
        eval_dataset=val_tokenized,
        processing_class=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_fn,
    )

    print("Training...")
    trainer.train()

    print("Saving model...")
    trainer.save_model(str(MODEL_DIR))
    tokenizer.save_pretrained(str(MODEL_DIR))

    # Save label mapping
    (MODEL_DIR / "labels.txt").write_text("\n".join(LABELS))

    print(f"\nDone. Model saved to {MODEL_DIR}")


if __name__ == "__main__":
    main()
