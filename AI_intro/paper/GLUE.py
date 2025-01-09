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
          "albert-base-v2", "google/electra-base-discriminator", "google/electra-small-discriminator"]
tasks = ["mnli", "qqp", "qnli", "sst2", "cola", "mrpc", "rte"]

pretrained_models = {
    "roberta-base": {
        "mnli": "TehranNLP-org/roberta-base-mnli-2e-5-42",                      # -0.5340856709158951, 0.8781456953642384
        "qqp": "TehranNLP-org/roberta-base-qqp-2e-5-42",                        # -0.5614161174849488, 0.9159040316596587
        "qnli": "textattack/roberta-base-QNLI",                                 # -0.5525178138247662, 0.9267801574226615
        "sst2": "textattack/roberta-base-SST-2",                                # -0.7155994137768582, 0.9403669724770642
        "cola": "textattack/roberta-base-CoLA",                                 # -0.40225478819813565, 0.6382594026155579
        "mrpc": "TehranNLP-org/roberta-base-mrpc-2e-5-42",                      # -0.3543709410468441, 0.8970588235294118
        "rte": "textattack/roberta-base-RTE",                                   # -0.6421491451004212, 0.7833935018050542
    },
    "distilroberta-base": {
        "mnli": None,
        "qqp": "Shobhank-iiitdwd/Distilroberta-base-QQP",                       # -0.738141810989102, 0.6318327974276527
        "qnli": "cross-encoder/qnli-distilroberta-base",                        # -0.8472257826613385, 0.4946000366099213
        "sst2": "aal2015/distilroberta-base-sst2-distilled",                    # -0.8194152449059582, 0.9277522935779816
        "cola": "mohammedbriman/distilroberta-base-finetuned-cola",             # -0.4375926363772261, 0.5788207437251082
        "mrpc": "EugenioRoma/distilroberta-base-mrpc-glue",                     # -0.6229590671981178, 0.8308823529411765
        "rte": None,
    },
    "distilbert-base-uncased": {
        "mnli": "typeform/distilbert-base-uncased-mnli",                        # -0.2205592725388641, 0.8211920529801324
        "qqp": "textattack/distilbert-base-uncased-QQP",                        # -0.9129112756344686, 0.5338609943111551
        "qnli": "textattack/distilbert-base-uncased-QNLI",                      # -0.4704686393263577, 0.8848617975471352
        "sst2": "distilbert/distilbert-base-uncased-finetuned-sst-2-english",   # -0.2630222378166057, 0.9105504587155964
        "cola": "textattack/distilbert-base-uncased-CoLA",                      # -0.792119084211421, 0.5685664296893979
        "mrpc": "textattack/distilbert-base-uncased-MRPC",                      # -0.5298636548772029, 0.8578431372549019
        "rte": "Vineetttt/distilbert-base-uncased-finetuned-rte",               # -0.833935266212873, 0.5992779783393501
    },
    "distilbert-base-cased": {
        "mnli": None,
        "qqp": "textattack/distilbert-base-cased-QQP",                          # -0.5754286202988488, 0.8974771209497897
        "qnli": None,   # "HeZhang1019/distilbert-base-cased-distilled-qnli-v1"
        "sst2": "textattack/distilbert-base-cased-SST-2",                       # -0.8326835243171337, 0.9002293577981652
        "cola": "textattack/distilbert-base-cased-CoLA",                        # -0.5758762985856327, 0.46372927911071965
        "mrpc": "textattack/distilbert-base-cased-MRPC",                        # -0.7982118975095636, 0.7843137254901961
        "rte": None,
    },
    "albert-base-v2": {
        "mnli": "Alireza1044/albert-base-v2-mnli",                              # -0.5617629793059487, 0.8448293428425879
        "qqp": "Alireza1044/albert-base-v2-qqp",                                # -0.6883012127418151, 0.9049715557754143
        "qnli": "Alireza1044/albert-base-v2-qnli",                              # -0.7711817913454362, 0.9136005857587406
        "sst2": "Alireza1044/albert-base-v2-sst2",                              # -0.6331057368447117, 0.9231651376146789
        "cola": "Alireza1044/albert-base-v2-cola",                              # -0.5233847261808, 0.5494768667363472
        "mrpc": "Alireza1044/albert-base-v2-mrpc",                              # -0.7048722692879975, 0.8627450980392157
        "rte": "Alireza1044/albert-base-v2-rte",                                # -1.0153694027610904, 0.6750902527075813
    },
    "google/electra-base-discriminator": {
        "mnli": "TehranNLP-org/electra-base-mnli",                              # -0.1270556164024973, 0.8874172185430463
        "qqp": "TehranNLP-org/electra-base-qqp-2e-5-42",                        # -0.41978923276613933, 0.9193420727182785
        "qnli": "cross-encoder/qnli-electra-base",                              # -0.8251125130076153, 0.4946000366099213
        "sst2": "TehranNLP-org/electra-base-sst2",                              # -0.5110292330649795, 0.9506880733944955
        "cola": "pszemraj/electra-base-discriminator-CoLA",                     # -0.4206984419701534, 0.6579677841732349
        "mrpc": "TehranNLP-org/electra-base-mrpc-2e-5-42",                      # -0.549591245533545, 0.8897058823529411
        "rte": "anirudh21/electra-base-discriminator-finetuned-rte",            # -0.7623099192650991, 0.8231046931407943
    },
    "google/electra-small-discriminator": {
        "mnli": "howey/electra-small-mnli",                                     # -0.4013931383553602, 0.8119205298013245
        "qqp": "howey/electra-small-qqp",                                       # -0.6306593387013033, 0.8947069008162256
        "qnli": None,
        "sst2": "Hazqeel/electra-small-finetuned-sst2",                         # -0.6847325826760177, 0.9174311926605505
        "cola": None,
        "mrpc": "Intel/electra-small-discriminator-mrpc",                       # -0.6972814945905685, 0.8529411764705882
        "rte": None,
    }
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
epoches = 3


def main(model_name: str, task_name: str):
    num_labels = 3 if task_name == "mnli" else 2
    pretrained_model_name = pretrained_models[model_name][task_name]
    if pretrained_model_name is None:
        return None, None
    print(f"Evaluating {model_name} on {task_name} task...")
    model = AutoModelForSequenceClassification.from_pretrained(pretrained_model_name)
    model.to(device)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name)
    # tokenizer.model_max_length = sys.maxsize
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    tokenized_datasets = get_dataset(task_name, tokenizer)
    eval_dataloader = DataLoader(
        tokenized_datasets["validation_matched"] if task_name == "mnli" else tokenized_datasets["validation"],
        shuffle=False, collate_fn=data_collator, batch_size=8, num_workers=4)
    metric = evaluate.load("glue", task_name)
    model.eval()
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
    count = 0
    for task_name in tasks[4:]:
        result = {}
        print(f"Task: {task_name}")
        for model_name in models:
            count += 1
            if count <= 4:
                continue
            score, accuracy = main(model_name, task_name)
            if score is None:
                continue
            result[model_name] = (score, accuracy)
            print(f"{model_name}: {score}, {accuracy}")
        print(result)
        print("-" * 50)