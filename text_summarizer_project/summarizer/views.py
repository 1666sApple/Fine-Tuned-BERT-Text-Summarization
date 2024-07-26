# summarizer/views.py

from django.shortcuts import render
from django.http import JsonResponse
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# Load the pre-trained model and tokenizer
tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")

def summarize_text(request):
    summary = ""
    if request.method == 'POST':
        input_text = request.POST.get('text', '')
        if input_text:
            inputs = tokenizer(input_text, max_length=1024, truncation=True, return_tensors="pt")
            summary_ids = model.generate(inputs["input_ids"], max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
            summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return render(request, 'summarizer/index.html', {'summary': summary})
