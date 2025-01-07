import os.path
import sys
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification, DataCollatorWithPadding, AdamW, get_scheduler
import datasets
import evaluate
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from LogME import LogME


models = ["roberta-base", "distilroberta-base", "distilbert-base-uncased", "distilbert-base-cased",
          "albert-base-v1", "albert-base-v2", "google/electra-base-discriminator", "google/electra-small-discriminator"]
tasks = ["mnli", "qqp", "qnli", "sst2", "cola", "mrpc", "rte"]

pretrained_models = {
    "roberta-base": {
        "mnli": "TehranNLP-org/roberta-base-mnli-2e-5-42",
        "qqp": "TehranNLP-org/roberta-base-qqp-2e-5-42",
        "qnli": "textattack/roberta-base-QNLI",
        "sst2": "textattack/roberta-base-SST-2",
        "cola": "textattack/roberta-base-CoLA",
        "mrpc": "TehranNLP-org/roberta-base-mrpc-2e-5-42",
        "rte": "textattack/roberta-base-RTE",
    },
    "distilroberta-base": {
        "mnli": "rambodazimi/distilroberta-base-finetuned-LoRA-MNLI",
        "qqp": "Shobhank-iiitdwd/Distilroberta-base-QQP",
        "qnli": "cross-encoder/qnli-distilroberta-base",
        "sst2": "azizbarank/distilroberta-base-sst-2-distilled",
        "cola": "mohammedbriman/distilroberta-base-finetuned-cola",
        "mrpc": "EugenioRoma/distilroberta-base-mrpc-glue",
        "rte": "rambodazimi/distilroberta-base-finetuned-LoRA-RTE",
    },
    "distilbert-base-uncased": {
        "mnli": "typeform/distilbert-base-uncased-mnli",
        "qqp": "textattack/distilbert-base-uncased-QQP",
        "qnli": "textattack/distilbert-base-uncased-QNLI",
        "sst2": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",
        "cola": "textattack/distilbert-base-uncased-CoLA",
        "mrpc": "textattack/distilbert-base-uncased-MRPC",
        "rte": "Vineetttt/distilbert-base-uncased-finetuned-rte",
    },
    "distilbert-base-cased": {
        "mnli": None,
        "qqp": "textattack/distilbert-base-cased-QQP",
        "qnli": "HeZhang1019/distilbert-base-cased-distilled-qnli-v1",
        "sst2": "textattack/distilbert-base-cased-SST-2",
        "cola": "textattack/distilbert-base-cased-CoLA",
        "mrpc": "textattack/distilbert-base-cased-MRPC",
        "rte": None,
    },
    "albert-base-v2": {
        "mnli": "Alireza1044/albert-base-v2-mnli",
        "qqp": "Alireza1044/albert-base-v2-qqp",
        "qnli": "Alireza1044/albert-base-v2-qnli",
        "sst2": "Alireza1044/albert-base-v2-sst2",
        "cola": "Alireza1044/albert-base-v2-cola",
        "mrpc": "Alireza1044/albert-base-v2-mrpc",
        "rte": "Alireza1044/albert-base-v2-rte",
    },
    "google/electra-base-discriminator": {
        "mnli": "TehranNLP-org/electra-base-mnli",
        "qqp": "TehranNLP-org/electra-base-qqp-2e-5-42",
        "qnli": "cross-encoder/qnli-electra-base",
        "sst2": "TehranNLP-org/electra-base-sst2",
        "cola": "pszemraj/electra-base-discriminator-CoLA",
        "mrpc": "TehranNLP-org/electra-base-mrpc-2e-5-42",
        "rte": "anirudh21/electra-base-discriminator-finetuned-rte",
    },
    "google/electra-small-discriminator": {
        "mnli": "howey/electra-small-mnli",
        "qqp": "howey/electra-small-qqp",
        "qnli": None,
        "sst2": "Hazqeel/electra-small-finetuned-sst2",
        "cola": "pszsemraj/electra-small-discriminator-CoLA",
        "mrpc": "Intel/electra-small-discriminator-mrpc",
        "rte": None,
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoches = 3


def main(model_name: str, task_name: str):
    # if os.path.exists(f"models/glue/fine_tuned_{model_name}_{task_name}.pth"):
    #     model = AutoModelForSequenceClassification.from_pretrained(model_name)
    #     model.load_state_dict(torch.load(f"models/glue/fine_tuned_{model_name}_{task_name}.pth"))
    #     model.to(device)
    # else:
    #     model = fine_tune(model_name, task_name)

    num_labels = 3 if task_name == "mnli" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    # tokenizer.model_max_length = sys.maxsize
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_dataset(task_name, tokenizer)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation_matched"] if task_name == "mnli" else tokenized_datasets["validation"],
        shuffle=False, collate_fn=data_collator, batch_size=8, num_workers=4)
    metric = evaluate.load("glue", task_name)
    model.eval()
    print(f"Evaluating {model_name} on {task_name} task...")
    F, Y = [], []
    for batch in eval_dataloader:
        batch = {k: v.to(device) for k, v in batch.items()}
        with torch.no_grad():
            outputs = model(**batch)
        logits = outputs.logits
        predictions = torch.argmax(logits, dim=-1)
        labels = batch["labels"]
        F.append(outputs.logits)
        Y.append(labels)
        metric.add_batch(predictions=predictions, references=labels)
    accuracy = metric.compute()["matthews_correlation"] if task_name == "cola" else metric.compute()["accuracy"]
    F = torch.cat([i for i in F])
    Y = torch.cat([i for i in Y])

    logme = LogME(is_regression=False)
    return logme.fit(F.numpy(), Y.numpy()), accuracy


def get_dataset(task_name: str, tokenizer):
    global tokenized_datasets
    raw_datasets = datasets.load_dataset('glue', task_name)
    # print(raw_datasets)
    if task_name == "mnli":                             # ['premise', 'hypothesis', 'label', 'idx']
        tokenized_datasets = raw_datasets.map(lambda examples: tokenizer(examples['premise'], examples['hypothesis']), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["premise", "hypothesis", "idx"])
    elif task_name == "qqp":                            # ['question1', 'question2', 'label', 'idx']
        tokenized_datasets = raw_datasets.map(lambda examples: tokenizer(examples['question1'], examples['question2']), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["question1", "question2", "idx"])
    elif task_name == "qnli":                           # ['question', 'sentence', 'label', 'idx']
        tokenized_datasets = raw_datasets.map(lambda examples: tokenizer(examples['question'], examples['sentence']), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["question", "sentence", "idx"])
    elif task_name == "sst2" or task_name == "cola":    # ['sentence', 'label', 'idx']
        tokenized_datasets = raw_datasets.map(lambda examples: tokenizer(examples['sentence']), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence", "idx"])
    elif task_name == "mrpc" or task_name == "rte":     # ['sentence1', 'sentence2', 'label', 'idx']
        tokenized_datasets = raw_datasets.map(lambda examples: tokenizer(examples['sentence1'], examples['sentence2']), batched=True)
        tokenized_datasets = tokenized_datasets.remove_columns(["sentence1", "sentence2", "idx"])
    tokenized_datasets = tokenized_datasets.rename_column("label", "labels")
    tokenized_datasets.set_format("torch")
    return tokenized_datasets


def fine_tune(model_name: str, task_name: str):
    num_labels = 3 if task_name == "mnli" else 2
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenized_datasets = get_dataset(task_name, tokenizer)
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    train_dataloader = DataLoader(tokenized_datasets["train"], shuffle=True, collate_fn=data_collator, batch_size=8,
                                  num_workers=4)

    optimizer = AdamW(model.parameters(), lr=5e-5)
    num_training_steps = len(train_dataloader) * epoches
    lr_scheduler = get_scheduler(
        name="linear",
        optimizer=optimizer,
        num_warmup_steps=0,
        num_training_steps=num_training_steps,
    )
    tqdm_bar = tqdm(range(num_training_steps))
    model.train()
    print(f"Fine-tuning {model_name} on {task_name} task...")
    min_loss = float('inf')
    for _ in range(epoches):
        train_loss = 0.
        for batch in train_dataloader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            loss = outputs.loss
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            tqdm_bar.update(1)
        train_loss /= len(train_dataloader)
        if train_loss < min_loss:
            min_loss = train_loss
            torch.save(model.state_dict(), f"models/glue/fine_tuned_{model_name}_{task_name}.pth")
            print(f"Model saved: fine_tuned_{model_name}_{task_name}.pth")
    return model


if __name__ == '__main__':
    for task_name in tasks[4:]:
        result = {}
        print(f"Task: {task_name}")
        for model_name in models:
            score, accuracy = main(model_name, task_name)
            result[model_name] = (score, accuracy)
            print(f"{model_name}: {score}, {accuracy}")
        print(result)
        print("-" * 50)