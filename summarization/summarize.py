# summarization/summarize.py

def summarize(blog_post, model, tokenizer):
    inputs = tokenizer(blog_post, max_length=1024, truncation=True, return_tensors="pt")
    summary_ids = model.generate(inputs["input_ids"], max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    summary = tokenizer.decode(summary_ids[0], skip_special_tokens=True)
    return summary
