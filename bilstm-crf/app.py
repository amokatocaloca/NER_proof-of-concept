# inference.py
import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from crf_layer import CRF
from data_utils import clean_token

nltk.download('punkt', quiet=True)
_ws_tok = WhitespaceTokenizer()

@st.cache_resource
def load_model_and_vocabs():
    try:
        model = load_model(
            "bilstm-crf/ner_model.keras",
            custom_objects={"CRF": CRF},
            compile=False
        )
        word2idx = np.load("bilstm-crf/word_vocab.npy", allow_pickle=True).item()
        tag2idx = np.load("bilstm-crf/tag_vocab.npy", allow_pickle=True).item()
        idx2tag = {v: k for k, v in tag2idx.items()}
        
        # Load rare labels used during training
        rare_labels = set(np.load("bilstm-crf/rare_labels.npy", allow_pickle=True).tolist())
        
        return model, word2idx, idx2tag, rare_labels
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        st.stop()

model, word2idx, idx2tag, rare_labels = load_model_and_vocabs()
MAX_LEN = model.input_shape[1]  # Original training max length

st.title("BiLSTM-CRF NER Tagger")
text = st.text_area("Enter text:", height=150, help="Supports long documents!")

def tokenize_text(text: str):
    raw_tokens = _ws_tok.tokenize(text)
    return [clean_token(tok).lower() for tok in raw_tokens]

def predict_long_sequence(tokens, window=100, stride=80):
    """Sliding window prediction for long sequences"""
    all_preds = []
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i+window]
        
        # Convert to indices
        unk_idx = word2idx.get("<UNK>")
        pad_idx = word2idx.get("<PAD>")
        ids = [word2idx.get(t, unk_idx) for t in chunk]
        
        # Pad to MAX_LEN or window size
        padded = ids + [pad_idx] * (MAX_LEN - len(ids)) if len(ids) < MAX_LEN else ids[:MAX_LEN]
        x = np.array([padded], dtype=np.int32)
        
        # Predict
        raw = model.predict(x, verbose=0)[0]
        pred_ids = np.argmax(raw, axis=-1)
        
        # Store predictions (only non-pad part)
        chunk_preds = [
            idx2tag.get(int(p_id), "O") 
            for p_id in pred_ids[:len(chunk)]
        ]
        all_preds.extend(chunk_preds)
    
    # Merge overlapping predictions
    return all_preds[:len(tokens)]

def postprocess_tags(tags):
    """Map rare labels to OTHER and clean IOB sequences"""
    cleaned = []
    prev_tag = "O"
    for tag in tags:
        # 1. Map rare tags to OTHER
        if tag in rare_labels:
            tag = "OTHER"
        
        # 2. Fix IOB inconsistencies
        if tag.startswith("I-") and prev_tag == "O":
            tag = "B" + tag[1:]  # Change I-X to B-X
        cleaned.append(tag)
        prev_tag = tag
    return cleaned

if text:
    tokens = tokenize_text(text)
    if not tokens:
        st.warning("No tokens after cleaning.")
        st.stop()
    
    if len(tokens) > MAX_LEN:
        st.warning(f"Truncated from {len(tokens)} to {MAX_LEN} tokens. Consider splitting into smaller sentences.")
    
    # Choose prediction method based on length
    if len(tokens) <= 2 * MAX_LEN:
        # Single prediction for shorter texts
        ids = [word2idx.get(t, word2idx["<UNK>"]) for t in tokens][:MAX_LEN]
        padded = ids + [word2idx["<PAD>"]] * (MAX_LEN - len(ids))
        x = np.array([padded], dtype=np.int32)
        raw = model.predict(x, verbose=0)[0]
        pred_ids = np.argmax(raw, axis=-1)
        tags = [idx2tag.get(int(i), "O") for i in pred_ids[:len(tokens)]]
    else:
        # Sliding window for long texts
        tags = predict_long_sequence(tokens)
    
    # Post-processing
    tags = postprocess_tags(tags)
    
    # Display results
    df = pd.DataFrame({"Token": tokens, "Tag": tags})
    
    # Highlight entities
    def color_entity(tag):
        if tag == "O": return ""
        color = "#E0FFFF" if "OTHER" in tag else "#E0FFFF"
        return f"background-color: {color}"
    
    st.dataframe(
        df.style.applymap(color_entity, subset=["Tag"]),
        use_container_width=True,
        height=400
    )