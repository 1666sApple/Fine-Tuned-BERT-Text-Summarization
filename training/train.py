# training/train.py

from transformers import TrainingArguments, Trainer

def preprocess(batch, tokenizer):
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

def train_model(model, tokenizer, train_dataset, val_dataset):
    train_dataset = train_dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
    val_dataset = val_dataset.map(lambda x: preprocess(x, tokenizer), batched=True)
    
    train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])
    val_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'labels'])

    training_args = TrainingArguments(
        output_dir="./results",
        per_device_train_batch_size=4,
        num_train_epochs=5,
        remove_unused_columns=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
    )

    trainer.train()
    return trainer
