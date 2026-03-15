from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForTokenClassification, TrainingArguments, Trainer
import evaluate
import numpy as np

# ─── Data loading ──────────────────────────────────────
dataset = load_dataset("json", data_files="labels.json")["train"]
dataset = dataset.train_test_split(test_size=0.15, seed=42)

tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")


def tokenize_and_align_labels(examples):
    tokenized_inputs = tokenizer(
        examples["tokens"],
        truncation=True,
        is_split_into_words=True,
        padding="max_length",
        max_length=128
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


tokenized_datasets = dataset.map(tokenize_and_align_labels, batched=True)

# ─── Model ──────────────────────────────────────────────────
id2label = {0: "O", 1: "B-ANIMAL"}
label2id = {"O": 0, "B-ANIMAL": 1}

model = AutoModelForTokenClassification.from_pretrained(
    "distilbert-base-uncased",
    num_labels=2,
    id2label=id2label,
    label2id=label2id
)

# ─── Training ──────────────────────────────────────────────
training_args = TrainingArguments(
    output_dir="./ner_animal_model",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=5,
    weight_decay=0.01,
    load_best_model_at_end=True,
    metric_for_best_model="f1",
    fp16=True,  # If GPU
)

metric = evaluate.load("seqeval")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)

    true_labels = [[id2label[l] for l in label if l != -100] for label in labels]
    pred_labels = [[id2label[p] for p, l in zip(prediction, label) if l != -100]
                   for prediction, label in zip(predictions, labels)]

    results = metric.compute(predictions=pred_labels, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_datasets["train"],
    eval_dataset=tokenized_datasets["test"],
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.save_model("ner_animal_final")
tokenizer.save_pretrained("ner_animal_final")