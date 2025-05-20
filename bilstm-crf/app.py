import re
import nltk
from nltk.tokenize import WhitespaceTokenizer
import streamlit as st
import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
from crf_layer import CRF
from data_utils import clean_token

# Download punkt for any NLTK-based sentence splitting
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
        return model, word2idx, idx2tag
    except Exception as e:
        st.error(f"Loading failed: {str(e)}")
        st.stop()

model, word2idx, idx2tag = load_model_and_vocabs()
MAX_LEN = model.input_shape[1]  # original training max sequence length

st.title("BiLSTM-CRF NER Tagger")
text = st.text_area("Enter text:", height=150, help="Supports long documents!")

def tokenize_text(text: str) -> list[str]:
    raw_tokens = _ws_tok.tokenize(text)
    return [clean_token(tok).lower() for tok in raw_tokens]


def predict_long_sequence(tokens: list[str], window: int = 100, stride: int = 80) -> list[str]:
    all_preds = []
    unk_idx = word2idx.get("<UNK>")
    pad_idx = word2idx.get("<PAD>")
    for i in range(0, len(tokens), stride):
        chunk = tokens[i:i + window]
        ids = [word2idx.get(t, unk_idx) for t in chunk]
        # pad or truncate to MAX_LEN
        padded = ids + [pad_idx] * (MAX_LEN - len(ids)) if len(ids) < MAX_LEN else ids[:MAX_LEN]
        x = np.array([padded], dtype=np.int32)
        raw = model.predict(x, verbose=0)[0]
        pred_ids = np.argmax(raw, axis=-1)
        chunk_preds = [idx2tag.get(int(p), "O") for p in pred_ids[: len(chunk)]]
        all_preds.extend(chunk_preds)
    return all_preds[: len(tokens)]


def postprocess_tags(tags: list[str]) -> list[str]:
    """
    Convert any I- tags that follow an O into B- tags.
    """
    cleaned: list[str] = []
    prev_tag = "O"
    for tag in tags:
        if tag.startswith("I-") and prev_tag == "O":
            # change I-X to B-X when previous tag was O
            label = tag.split('-', 1)[1]
            tag = f"B-{label}"
        cleaned.append(tag)
        prev_tag = tag
    return cleaned

if text:
    tokens = tokenize_text(text)
    if not tokens:
        st.warning("No tokens after cleaning.")
        st.stop()

    if len(tokens) > MAX_LEN:
        st.warning(
            f"Truncated from {len(tokens)} to {MAX_LEN} tokens. "
            "Consider splitting into smaller chunks."
        )

    # choose prediction strategy
    if len(tokens) <= 2 * MAX_LEN:
        unk_idx = word2idx.get("<UNK>")
        pad_idx = word2idx.get("<PAD>")
        ids = [word2idx.get(t, unk_idx) for t in tokens][:MAX_LEN]
        padded = ids + [pad_idx] * (MAX_LEN - len(ids))
        x = np.array([padded], dtype=np.int32)
        raw = model.predict(x, verbose=0)[0]
        pred_ids = np.argmax(raw, axis=-1)
        tags = [idx2tag.get(int(i), "O") for i in pred_ids[: len(tokens)]]
    else:
        tags = predict_long_sequence(tokens)

    # post-process
    tags = postprocess_tags(tags)

    # prepare DataFrame
    df = pd.DataFrame({"Token": tokens, "Tag": tags})

    # styling
    def color_entity(tag: str) -> str:
        return "background-color: #E0FFFF" if tag != "O" else ""

    st.dataframe(
        df.style.applymap(color_entity, subset=["Tag"]),
        use_container_width=True,
        height=400,
    )
