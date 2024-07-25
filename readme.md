# Text Summarizer

## Overview

This project aims to build a text summarization model using the BART (Bidirectional and Auto-Regressive Transformers) model. The summarizer is trained on the DialogSum dataset to generate concise and coherent summaries of dialogues.

## Project Structure

The project is organized as follows:

```bash
text_summarizer/
├── data/
│   ├── train.csv
│   ├── validation.csv
│   ├── test.csv
├── models/
│   ├── __init__.py
│   ├── model.py
├── preprocessing/
│   ├── __init__.py
│   ├── data_preparation.py
├── training/
│   ├── __init__.py
│   ├── train.py
├── summarization/
│   ├── __init__.py
│   ├── summarize.py
├── utils/
│   ├── __init__.py
│   ├── setup.py
├── requirements.txt
├── main.py
└── README.md
```


### Directories and Files

- **data/**: Contains the dataset files.
- **models/**: Contains code related to model loading and saving.
- **preprocessing/**: Contains code for data loading and preparation.
- **training/**: Contains code for training the model.
- **summarization/**: Contains code for generating summaries.
- **utils/**: Contains utility scripts, such as setup scripts.
- **main.py**: Main script to run the training and summarization process.
- **requirements.txt**: List of required Python packages.

## Setup

### Prerequisites

Ensure you have Python installed (Python 3.10+ recommended). You can download it from [python.org](https://www.python.org/).

### Installation

1. **Clone the repository:**
    ```bash
    git clone https://github.com/1666sApple/Fine-Tuned-BERT-Text-Summarization.git
    cd Fine-Tuned-BERT-Text-Summarization
    ```

2. **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```

### Directory Setup

Before running the scripts, ensure the necessary directories are set up:

```bash
python3 -c "from utils.setup import setup_directories; setup_directories()"
```

## Usage

### Data Preparation

1. **Load and save datasets:**
    ```bash
    python3 -c "from preprocessing.data_preparation import load_and_save_datasets; load_and_save_datasets()"
    ```
2. **Load datasets:**
    ```bash
    python3 run_datasets.py
    ```
### Training the model

To train the model, run the following command:

```python
python3 main.py
```

**This will:**

- Load and save the datasets.
- Load the BART model and tokenizer.
- Train the model on the training dataset.
- Evaluate the model on the validation dataset.
- Save the trained model and tokenizer.

### Summarization

To generate summaries using the trained model, you can use the `summarize` function in the `summarization/summarize.py` script. Here's an example:

```python
from summarization.summarize import summarize
from models.model import load_saved_model_and_tokenizer

# Load the saved model and tokenizer
tokenizer, model = load_saved_model_and_tokenizer('/content/model/')

# Summarize a sample dialogue
sample_dialogue = """Put any sample dialogue here."""
summary = summarize(sample_dialogue, model, tokenizer)
print("Summary:", summary)
```

## Model and Tokenizer

### Loading Pre-Trained Models

The pre-trained BART model and tokenizer are loaded from Hugging Face's model hub:

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-large-cnn")
```

### Saving and Loading Trained Models

After training, the model and tokenizer can be saved and loaded as follows:

```python
from models.model import save_model_and_tokenizer, load_saved_model_and_tokenizer

# Save the model and tokenizer
save_model_and_tokenizer(model, tokenizer, '/content/model/')

# Load the saved model and tokenizer
tokenizer, model = load_saved_model_and_tokenizer('/content/model/')
```

## Additional Information

### Preprocessing

The preprocessing step includes tokenizing the dialogue and summary texts using the BART tokenizer. It also formats the dataset to be compatible with PyTorch.

### Training

The training is conducted using Hugging Face's Trainer class, which simplifies the training loop and integrates seamlessly with the Transformers library.

### Evaluation

After training, the model is evaluated on the validation dataset to measure its performance.

## Contributing

Contributions are welcome! Please create a pull request with a detailed description of your changes.

## Contact

Contact
For any questions or issues, please contact Ananno Asif at [ananno.034@gmail.com].

