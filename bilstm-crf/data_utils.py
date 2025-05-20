# data_utils.py

import pandas as pd
import re
import os
import numpy as np
from collections import Counter

# Special tokens
UNK = "$UNK$"
NUM = "$NUM$"
NONE = "O"

# --- Custom Exception ---
class MyIOError(Exception):
    def __init__(self, filename):
        message = f"""
ERROR: Unable to locate file {filename}.

FIX: Have you tried running your data build script first?
This script should build the vocabulary from your train, dev, and test sets,
and trim your word vectors accordingly.
"""
        super(MyIOError, self).__init__(message)


# --- Token Cleaning ---
def clean_token(token):
    """
    Cleans a token by removing extraneous whitespace and surrounding quotes.
    Optionally, you can remove unwanted punctuation by uncommenting the regex line.
    """
    token = token.strip()
    token = token.lstrip('"').rstrip('"').lstrip("'").rstrip("'")
    return token


# --- Data Loading and Vocabulary Building ---
def load_csv_data(file_path):
    """
    Load a CSV file and apply token cleaning.
    Assumes the CSV has columns: doc_id, token, bio_label.

    Returns:
        data: the raw dataframe
        grouped: a dataframe grouped by doc_id, with columns:
                 - token: list of tokens for that document
                 - bio_label: list of corresponding labels
    """
    if not os.path.isfile(file_path):
        raise MyIOError(file_path)

    data = pd.read_csv(file_path)
    data['token'] = data['token'].apply(clean_token)
    grouped = (
        data
        .groupby('doc_id')
        .agg({'token': list, 'bio_label': list})
        .reset_index()
    )
    return data, grouped


