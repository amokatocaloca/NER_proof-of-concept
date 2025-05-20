import os
import random
import numpy as np
import tensorflow as tf
from ner_model import build_and_train_bilstm_crf
from sklearn.metrics import f1_score
from seqeval.metrics import f1_score as seq_f1_score
from seqeval.metrics import classification_report



# 1) Seed Python, NumPy, TensorFlow, and the built-in random
seed = 42
random.seed(seed)
np.random.seed(seed)
tf.random.set_seed(seed)

def calculate_per_class_f1(model, X_test, y_test, tag_vocab):
    y_pred = model.predict(X_test)
    # Convert the continuous outputs to discrete labels.
    y_pred = np.argmax(y_pred, axis=-1) 
    
    pad_idx = tag_vocab["<PAD>"]
    mask = y_test != pad_idx
    return f1_score(y_test[mask], y_pred[mask], average=None)

def ner_classification_report(model, X_test, y_test, tag_vocab):
    y_pred = model.predict(X_test)
    y_pred = np.argmax(y_pred, axis=-1)

    # Reverse the vocab
    id2tag = {v: k for k, v in tag_vocab.items()}
    pad_idx = tag_vocab["<PAD>"]

    true_tags = []
    pred_tags = []

    for seq_true, seq_pred in zip(y_test, y_pred):
        valid = np.where(seq_true != pad_idx)[0]
        true_tags.append(
            [ id2tag.get(idx, "O") for idx in seq_true[valid] ]
        )
        pred_tags.append(
            [ id2tag.get(idx, "O") for idx in seq_pred[valid] ]
        )
    
    return classification_report(true_tags, pred_tags)


if __name__ == "__main__":
    data = np.load("bilstm-crf/split_data.npz", allow_pickle=True)
    X_train, y_train = data["X_train"], data["y_train"]
    X_val,   y_val   = data["X_val"],   data["y_val"]
    X_test,  y_test  = data["X_test"],  data["y_test"]
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}")
    

    

    embedding_matrix = np.load("bilstm-crf/embeddings.npz")["embeddings"]
    print("Embedding matrix shape:", embedding_matrix.shape)  # Should match (vocab_size, 300
    vocab_size = embedding_matrix.shape[0]
    tag_vocab = np.load("bilstm-crf/tag_vocab.npy", allow_pickle=True).item()

    pad_idx = tag_vocab["<PAD>"]
    flat_labels = y_train[y_train != pad_idx]
    classes = np.unique(flat_labels)

    inferred_num_tags = int(y_train.max()) + 1
    print("→ inferring num_tags =", inferred_num_tags)


    model, history, test_loss = build_and_train_bilstm_crf(
        X_train, y_train,
        X_test, y_test,
        X_val,   y_val,        
        max_len=X_train.shape[1],
        vocab_size=vocab_size,
        num_tags=inferred_num_tags,      
        embedding_matrix=embedding_matrix,
        lstm_units=64,
        lstm_dropout=0.2,
        epochs=50,
        batch_size=32
    )

    model.save("bilstm-crf/ner_model.keras")
    
    id2tag = {v: k for k, v in tag_vocab.items()}


   # ─── Validation Set Evaluation ────────────────────────────────────────
    print("\n=== Validation Set Evaluation ===")
    y_val_pred_raw = model.predict(X_val, batch_size=32, verbose=1)
    y_val_pred = np.argmax(y_val_pred_raw, axis=-1)

    # build true/pred lists for seqeval
    true_tags_val, pred_tags_val = [], []
    for seq_true, seq_pred in zip(y_val, y_val_pred):
        valid = np.where(seq_true != pad_idx)[0]
        true_tags_val.append([ id2tag[idx] for idx in seq_true[valid] ])
        pred_tags_val.append([ id2tag[idx] for idx in seq_pred[valid] ])

    # overall F1
    val_entity_f1 = seq_f1_score(true_tags_val, pred_tags_val)
    print(f"Validation entity‐level F1: {val_entity_f1:.4f}")

    # per‐class report
    print("\n=== Validation Classification Report ===")
    print(classification_report(true_tags_val, pred_tags_val))

    # ─── Test Set Evaluation ───────────────────────────────────────────────
    print("\n=== Test Set Evaluation ===")
    y_test_pred_raw = model.predict(X_test, batch_size=32, verbose=1)
    y_test_pred = np.argmax(y_test_pred_raw, axis=-1)

    true_tags_test, pred_tags_test = [], []
    for seq_true, seq_pred in zip(y_test, y_test_pred):
        valid = np.where(seq_true != pad_idx)[0]
        true_tags_test.append([ id2tag[idx] for idx in seq_true[valid] ])
        pred_tags_test.append([ id2tag[idx] for idx in seq_pred[valid] ])

    test_entity_f1 = seq_f1_score(true_tags_test, pred_tags_test)
    print(f"Test entity‐level F1:      {test_entity_f1:.4f}")

    print("\n=== Test Classification Report ===")
    print(classification_report(true_tags_test, pred_tags_test))
  