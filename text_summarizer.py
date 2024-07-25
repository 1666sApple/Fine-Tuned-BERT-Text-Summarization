#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# In[2]:


# Define the splits and file paths
splits = {
    'train': 'train.csv',
    'validation': 'validation.csv',
    'test': 'test.csv'
}

# Loop through each split, load the DataFrame, and save it to the current directory
for split_name, file_name in splits.items():
    # Load the DataFrame from the specified file
    df = pd.read_csv("hf://datasets/knkarthick/dialogsum/" + file_name)
    
    # Save the DataFrame to the current directory with a new name
    df.to_csv(f'{split_name}_data.csv', index=False)


# In[3]:


train_df = pd.read_csv("data/train_data.csv")
val_df = pd.read_csv("data/validation_data.csv")
test_df = pd.read_csv("data/test_data.csv")

train_df.head()


# In[4]:


from datasets import Dataset, DatasetDict

# Convert DataFrames to Datasets
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)
test_dataset = Dataset.from_pandas(test_df)


# In[5]:


# Load model directly
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")


# In[6]:


# Tokenization function
def preprocess(batch):
    source = batch['dialogue']
    target = batch['summary']
    source_ids = tokenizer(source, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    target_ids = tokenizer(target, padding='max_length', truncation=True, max_length=512, return_tensors='pt')
    
    labels = target_ids['input_ids']
    labels[labels == tokenizer.pad_token_id] = -100

    return {
        'input_ids': source_ids['input_ids'].squeeze(),
        'attention_mask': source_ids['attention_mask'].squeeze(),
        'labels': labels.squeeze()
    }


# In[7]:


# Apply preprocessing to datasets
train_dataset = train_dataset.map(preprocess, batched=True)
val_dataset = val_dataset.map(preprocess, batched=True)
test_dataset = test_dataset.map(preprocess, batched=True)


# In[8]:


# Remove unnecessary columns
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])


# In[9]:


from transformers import TrainingArguments, Trainer

# Training arguments
training_args = TrainingArguments(
    output_dir="./results",
    per_device_train_batch_size=4,
    num_train_epochs=5,
    remove_unused_columns=True,
)
    


# In[10]:


# Initialize the Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
)


# In[11]:


trainer.train()


# In[ ]:


eval_results = trainer.evaluate()


# In[ ]:


model.save_pretrained('/content/model/')
tokenizer.save_pretrained('/content/model/')


# In[ ]:


tokenizer = AutoTokenizer.from_pretrained("/content/model/")
model = AutoModelForSeq2SeqLM.from_pretrained("/content/model/")


# In[ ]:


def summarize(blog_post):
    #Tokenize input blog post
    inputs = tokenizer(blog_post, max_length=1024, truncation=True, return_tensors="pt")
    
    # Generate Summary
    summary_idx = model.generate(inputs["input_ids"], max_length=200, min_length=50, length_penalty=2.0, num_beams=4, early_stopping=True)
    
    # Decode the summary
    summary = tokenizer.decode(summary_idx[0], skip_special_tokens=True)
    
    return summary
    

