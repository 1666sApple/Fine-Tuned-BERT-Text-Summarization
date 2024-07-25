# models/model.py

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

def load_model_and_tokenizer():
    tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
    return tokenizer, model

def save_model_and_tokenizer(model, tokenizer, save_path):
    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

def load_saved_model_and_tokenizer(load_path):
    tokenizer = AutoTokenizer.from_pretrained(load_path)
    model = AutoModelForSeq2SeqLM.from_pretrained(load_path)
    return tokenizer, model
