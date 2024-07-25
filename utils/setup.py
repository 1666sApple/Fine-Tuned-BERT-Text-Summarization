# utils/setup.py

import os

def setup_directories():
    directories = ["data", "models", "preprocessing", "training", "summarization", "utils"]
    for directory in directories:
        if not os.path.exists(directory):
            os.makedirs(directory)
