# main.py

from preprocessing.data_preparation import load_and_save_datasets, load_datasets
from models.model import load_model_and_tokenizer, save_model_and_tokenizer, load_saved_model_and_tokenizer
from training.train import train_model
from summarization.summarize import summarize
from utils.setup import setup_directories

def main():
    # Setup directories
    setup_directories()

    # Load and save datasets
    load_and_save_datasets()
    
    # Load datasets
    train_dataset, val_dataset, test_dataset = load_datasets()
    
    # Load model and tokenizer
    tokenizer, model = load_model_and_tokenizer()
    
    # Train model
    trainer = train_model(model, tokenizer, train_dataset, val_dataset)
    
    # Save model and tokenizer
    save_model_and_tokenizer(model, tokenizer, '/content/model/')
    
    # Load the saved model and tokenizer
    tokenizer, model = load_saved_model_and_tokenizer('/content/model/')
    
    # Summarize a sample dialogue
    sample_dialogue = train_dataset[0]['dialogue']
    summary = summarize(sample_dialogue, model, tokenizer)
    print("Sample Dialogue:", sample_dialogue)
    print("Summary:", summary)

if __name__ == "__main__":
    main()
