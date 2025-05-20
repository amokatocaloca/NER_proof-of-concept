import streamlit as st
import torch
import joblib
from transformers import BertTokenizerFast, BertForTokenClassification
import pandas as pd

# Load model and components
@st.cache_resource
def load_model():
    model = BertForTokenClassification.from_pretrained("bert/results/checkpoint-45")
    tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', add_prefix_space=True)
    le = joblib.load("bert/label_encoder.joblib")
    return model, tokenizer, le

model, tokenizer, le = load_model()
id2label = model.config.id2label

def predict_ner(text):
    tokens = text.split()
    encoding = tokenizer(
        tokens,
        is_split_into_words=True,
        return_tensors="pt",
        truncation=True,
        max_length=128
    )
    
    # Predict
    with torch.no_grad():
        outputs = model(**encoding)
    
    # Get predictions and map to labels
    predictions = torch.argmax(outputs.logits, dim=2).squeeze().numpy()
    word_ids = encoding.word_ids()
    
    # Map predictions to original words
    current_word = None
    results = []
    for idx, word_id in enumerate(word_ids):
        if word_id is None or word_id == current_word:
            continue
        current_word = word_id
        results.append((tokens[word_id], id2label[predictions[idx]]))
    
    return results

# Streamlit UI
st.title("BERT NER Tagger")
text_input = st.text_input("Enter a sentence:")

if text_input:
    preds = predict_ner(text_input)
    # st.write(f"⚠️ OOV rate: {oov:.0%}")
    if not preds:
        st.warning("No valid tokens.")
    else:
        df = pd.DataFrame(preds, columns=["Token", "Label"])
        st.dataframe(df, use_container_width=True)