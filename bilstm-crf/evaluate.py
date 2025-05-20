# evaluate_full.py
import re
import time
import psutil
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from difflib import SequenceMatcher
from tensorflow.keras.models import load_model
from crf_layer import CRF
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score, f1_score as entity_f1_score

# ─── PARAMETERS ────────────────────────────────────────────────────────
CSV_PATH    = "bilstm-crf/bio_output.csv"
MODEL_PATH  = "bilstm-crf/ner_model.keras"
WORD_VOCAB  = "bilstm-crf/word_vocab.npy"
TAG_VOCAB   = "bilstm-crf/tag_vocab.npy"
SPLIT_DATA  = "bilstm-crf/split_data.npz"  # to get max_len
BATCH_SIZE  = 32

def load_vocab(path):
    return np.load(path, allow_pickle=True).item()

# ─── 1) Load & clean full CSV ───────────────────────────────────────────
df = pd.read_csv(CSV_PATH)
df = df[df.token.notna()].copy()
df.token = df.token.str.strip()
df = df[df.token != ""]

# group into (tokens, labels)
grp = df.groupby("doc_id").agg({
    "token":     lambda s: list(s),
    "bio_label": lambda s: list(s)
}).reset_index()

# ─── 2) Load vocabs & define special IDs ────────────────────────────────
word_vocab = load_vocab(WORD_VOCAB)
tag_vocab  = load_vocab(TAG_VOCAB)
UNK    = word_vocab.get("<UNK>")
PAD_W  = word_vocab["<PAD>"]
PAD_T  = tag_vocab["<PAD>"]
# reverse map for decoding
id2tag = {i: t for t, i in tag_vocab.items()}

# get max_len from your split_data.npz
spl     = np.load(SPLIT_DATA, allow_pickle=True)
max_len = spl["X_train"].shape[1]

# ─── 3) Build X_all, y_all ─────────────────────────────────────────────
X_all, y_all = [], []
for toks, labs in zip(grp.token, grp.bio_label):
    # tokens → ids
    x = [word_vocab.get(t, UNK) for t in toks]
    # BIO labels → ids, fallback to 'O' if unseen
    y = [tag_vocab.get(l, tag_vocab["O"]) for l in labs]

    # pad or truncate to max_len
    if len(x) > max_len:
        x, y = x[:max_len], y[:max_len]
    else:
        x = x + [PAD_W] * (max_len - len(x))
        y = y + [PAD_T] * (max_len - len(y))

    X_all.append(x)
    y_all.append(y)

X_all = np.array(X_all, dtype=np.int32)
y_all = np.array(y_all, dtype=np.int32)

# ─── 4) Load model & predict ────────────────────────────────────────────
print("Loading model…")
model = load_model(MODEL_PATH, custom_objects={"CRF": CRF}, compile=False)

print("Running inference on full dataset…")
t0 = time.time()
y_pred_raw = model.predict(X_all, batch_size=BATCH_SIZE, verbose=1)
t1 = time.time()
print(f"Inference time: {t1-t0:.2f}s")
print(f"Memory RSS: {psutil.Process().memory_info().rss/1024**2:.1f} MB")

# ─── 5) Decode predictions ─────────────────────────────────────────────
y_pred_ids = np.argmax(y_pred_raw, axis=-1)

true_tags, pred_tags = [], []
for true_seq, pred_seq in zip(y_all, y_pred_ids):
    t_seq, p_seq = [], []
    for t_id, p_id in zip(true_seq, pred_seq):
        if t_id == PAD_T:
            break
        t_seq.append(id2tag[t_id])
        p_seq.append(id2tag[p_id])
    true_tags.append(t_seq)
    pred_tags.append(p_seq)

# ─── 6) Entity-level report ────────────────────────────────────────────
print("\n=== Entity-level Classification Report ===")
print(classification_report(true_tags, pred_tags))

entity_f1 = entity_f1_score(true_tags, pred_tags)
print(f"Overall entity-level F1:         {entity_f1:.4f}")

# ─── 7) MUC-5 & SPU metrics (unchanged) ─────────────────────────────────
def compute_muc5(gold, pred):
    COR = INC = MIS = SPU = 0
    for g_seq, p_seq in zip(gold, pred):
        for g, p in zip(g_seq, p_seq):
            if   g == p:          COR += 1
            elif g != 'O' and p == 'O': MIS += 1
            elif g == 'O' and p != 'O': SPU += 1
            else:                  INC += 1
    return COR, INC, MIS, SPU

COR, INC, MIS, SPU = compute_muc5(true_tags, pred_tags)
print("\nMUC-5 Token Counts:")
print(f"  COR: {COR}, INC: {INC}, MIS: {MIS}, SPU: {SPU}")

def split_spu(tokens_seqs, gold_seqs, pred_seqs, thresh=0.90):
    crit = noncrit = 0
    for toks, gold, pred in zip(tokens_seqs, gold_seqs, pred_seqs):
        sp_idx = [i for i,(g,p) in enumerate(zip(gold,pred)) if g=="O" and p!="O"]
        if not sp_idx:
            continue

        span = " ".join(toks[i] for i in sp_idx)
        if re.search(r"\d", span):
            crit += len(sp_idx)
        else:
            sent = " ".join(toks)
            sim = SequenceMatcher(None, sent, sent.replace(span, "")).ratio()
            if sim < thresh:
                crit += len(sp_idx)
            else:
                noncrit += len(sp_idx)

    return crit, noncrit

id2word = {i: w for w, i in word_vocab.items()}
tokens_seqs = []
for seq in X_all:
    toks = []
    for idx in seq:
        if idx == PAD_W: break
        toks.append(id2word.get(idx, "<UNK>"))
    tokens_seqs.append(toks)

crit, noncrit = split_spu(tokens_seqs, true_tags, pred_tags)
print(f"\nSPU breakdown: Critical={crit}, Non-critical={noncrit}")

# ─── 8) Plot & save ────────────────────────────────────────────────────
pr = precision_score(true_tags, pred_tags)
rc = recall_score(true_tags, pred_tags)
f1 = entity_f1
plt.figure(); plt.bar(["P","R","F1"], [pr, rc, f1]); plt.ylim(0,1)
plt.title("Overall NER Metrics"); plt.savefig("metrics_full.png"); plt.close()

print("\n→ Saved:  muc5_full.png, spu_full.png, metrics_full.png")
