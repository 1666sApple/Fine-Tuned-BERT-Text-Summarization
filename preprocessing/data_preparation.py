# preprocessing/data_preparation.py

import pandas as pd
from datasets import Dataset

def load_and_save_datasets():
    splits = {
        'train': 'train.csv',
        'validation': 'validation.csv',
        'test': 'test.csv'
    }

    for split_name, file_name in splits.items():
        df = pd.read_csv("hf://datasets/knkarthick/dialogsum/" + file_name)
        df.to_csv(f'data/{split_name}_data.csv', index=False)

def load_datasets():
    train_df = pd.read_csv("data/train_data.csv")
    val_df = pd.read_csv("data/validation_data.csv")
    test_df = pd.read_csv("data/test_data.csv")

    train_dataset = Dataset.from_pandas(train_df)
    val_dataset = Dataset.from_pandas(val_df)
    test_dataset = Dataset.from_pandas(test_df)

    return train_dataset, val_dataset, test_dataset
