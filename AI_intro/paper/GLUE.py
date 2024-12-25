from transformers import RobertaModel, DistilBertModel, AlbertModel, ElectraForPreTraining, RobertaTokenizer, \
    DistilBertTokenizer, AlbertTokenizer, ElectraTokenizerFast
import pandas as pd


models = ["roberta", "distil-roberta", "distilbert-uncased", "distilbert-cased", "albert-v1", "albert-v2", "electra-base", "electra-small"]
tasks = ["MNLI", "QQP", "QNLI", "SST-2", "CoLA", "MRPC", "RTE"]

data_path = "E:/DataSets/GLUE/"


def get_model(model_name:str):
    if model_name == "roberta":
        return RobertaModel.from_pretrained("roberta-base"), RobertaTokenizer.from_pretrained("roberta-base")
    elif model_name == "distil-roberta":
        return DistilBertModel.from_pretrained('distilroberta-base'), DistilBertTokenizer.from_pretrained('distilroberta-base')
    elif model_name == "distilbert-uncased":
        return DistilBertModel.from_pretrained('distilbert-base-uncased'), DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
    elif model_name == "distilbert-cased":
        return DistilBertModel.from_pretrained('distilbert-base-cased'), DistilBertTokenizer.from_pretrained('distilbert-base-cased')
    elif model_name == "albert-v1":
        return AlbertModel.from_pretrained('albert-base-v1'), AlbertTokenizer.from_pretrained('albert-base-v1')
    elif model_name == "albert-v2":
        return AlbertModel.from_pretrained('albert-base-v2'), AlbertTokenizer.from_pretrained('albert-base-v2')
    elif model_name == "electra-base":
        return ElectraForPreTraining.from_pretrained('google/electra-base-discriminator'), ElectraTokenizerFast.from_pretrained('google/electra-base-discriminator')
    elif model_name == "electra-small":
        return ElectraForPreTraining.from_pretrained('google/electra-small-discriminator'), ElectraTokenizerFast.from_pretrained('google/electra-small-discriminator')
