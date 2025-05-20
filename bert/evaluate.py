import re
import time
import joblib
import numpy as np
import pandas as pd
from difflib import SequenceMatcher
import torch
import psutil
import matplotlib.pyplot as plt
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    Trainer,
    TrainingArguments,
    DataCollatorForTokenClassification
)
from seqeval.metrics import classification_report, precision_score, recall_score, f1_score
from train_bert import NERDataset  # adjust import if your dataset class is elsewhere

# Disable tokenizers parallelism warning
TOKENIZERS_PARALLELISM = False

# --- Parameters (customize paths) ---
TEST_CSV       = "bert/data/bio_output.csv"
MODEL_DIR      = "bert/results/checkpoint-45"
LABEL_ENCODER  = "bert/label_encoder.joblib"
BATCH_SIZE     = 16

# ─── Helper: split_spu ───────────────────────────────────────────────
def split_spu(token_seqs, true_labels, pred_labels, similarity_thresh=0.90):
    crit = noncrit = 0
    for tokens, gold, pred in zip(token_seqs, true_labels, pred_labels):
        sent_text = " ".join(tokens)
        spurious = [tok for tok, g, p in zip(tokens, gold, pred)
                    if g == "OTHER" and p != "OTHER"]
        if not spurious:
            continue
        spu_text = " ".join(spurious)
        if re.search(r"\d", spu_text):
            crit += 1
            continue
        sim = SequenceMatcher(None, sent_text, sent_text.replace(spu_text, "")).ratio()
        if sim < similarity_thresh:
            crit += 1
        else:
            noncrit += 1
    return crit, noncrit

# ─── Helper: compute MUC-5 token metrics ─────────────────────────────
def compute_muc5(true_labels, pred_labels):
    COR = INC = MIS = SPU = 0
    for gold_seq, pred_seq in zip(true_labels, pred_labels):
        for g, p in zip(gold_seq, pred_seq):
            if g == p:
                COR += 1
            elif g != 'OTHER' and p == 'OTHER':
                MIS += 1
            elif g == 'OTHER' and p != 'OTHER':
                SPU += 1
            else:
                INC += 1
    return COR, INC, MIS, SPU

# 1) Load and preprocess raw token-level CSV
df_raw = pd.read_csv(TEST_CSV)
for col in ('doc_id', 'token', 'bio_label'):
    if col not in df_raw.columns:
        raise ValueError(f"Column '{col}' missing in {TEST_CSV}")
df_raw = df_raw[df_raw['token'].notna()].copy()
df_raw['token'] = df_raw['token'].str.strip()
df_raw = df_raw[df_raw['token'] != '']

docs = df_raw.groupby('doc_id').agg({
    'token': lambda s: list(s),
    'bio_label': lambda s: list(s)
}).reset_index().rename(columns={'token':'tokens','bio_label':'bio_labels'})

# 2) Merge labels based on LabelEncoder classes
le = joblib.load(LABEL_ENCODER)
valid_labels = set(le.classes_)
docs['bio_labels_merged'] = docs['bio_labels']

# 3) Prepare Dataset and load trained model
docs['encoded_labels'] = docs['bio_labels_merged'].apply(lambda labs: le.transform(labs))
tokenizer = BertTokenizerFast.from_pretrained(MODEL_DIR)
test_dataset  = NERDataset(docs['tokens'], docs['encoded_labels'], tokenizer)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = BertForTokenClassification.from_pretrained(
    MODEL_DIR,
    id2label={i: lbl for i, lbl in enumerate(le.classes_)},
    label2id={lbl: i for i, lbl in enumerate(le.classes_)}
).to(device)

# 4) Setup Trainer for inference only
trainer = Trainer(
    model=model,
    args=TrainingArguments(
        output_dir='./tmp_eval',
        per_device_eval_batch_size=BATCH_SIZE,
        logging_steps=5
    ),
    data_collator=DataCollatorForTokenClassification(tokenizer)
)

# 5) Profile inference: time, CPU & GPU memory
print('Profiling inference...')
# Reset GPU peak memory stats
if torch.cuda.is_available():
    torch.cuda.reset_peak_memory_stats(device)

start_time = time.time()
out = trainer.predict(test_dataset)
end_time = time.time()

elapsed = end_time - start_time
cpu_mem = psutil.Process().memory_info().rss / (1024**2)  # in MB
if torch.cuda.is_available():
    gpu_peak = torch.cuda.max_memory_allocated(device) / (1024**3)  # in GB
else:
    gpu_peak = None

print(f"Inference time: {elapsed:.2f}s")
print(f"CPU memory (RSS): {cpu_mem:.1f} MB")
if gpu_peak is not None:
    print(f"GPU peak memory: {gpu_peak:.2f} GB")

pred_ids = np.argmax(out.predictions, axis=2)
label_ids = out.label_ids

# 6) Reconstruct true/pred label sequences
token_seqs  = docs['tokens'].tolist()
true_labels = docs['bio_labels_merged'].tolist()
pred_labels = []
for p_seq, g_seq in zip(pred_ids, label_ids):
    seq_preds = []
    for p_id, g_id in zip(p_seq, g_seq):
        if g_id == -100:
            continue
        seq_preds.append(le.classes_[p_id])
    pred_labels.append(seq_preds)

# 7) Print classification report
print('\nClassification Report:')
print(classification_report(true_labels, pred_labels))
print(f"F1 Score: {f1_score(true_labels, pred_labels):.4f}")

# 8) Compute and print MUC-5 metrics
COR, INC, MIS, SPU = compute_muc5(true_labels, pred_labels)
print(f"\nMUC-5 Token Counts:")
print(f"  Correct (COR):  {COR}")
print(f"  Incorrect (INC):{INC}")
print(f"  Missing (MIS):  {MIS}")
print(f"  Spurious (SPU): {SPU}")

# 9) SPU error breakdown
crit, noncrit = split_spu(token_seqs, true_labels, pred_labels)
print(f"\nCritical SPU errors:     {crit}")
print(f"Non-critical SPU errors: {noncrit}")

# 10) Plot Precision, Recall, F1
pr = precision_score(true_labels, pred_labels)
rc = recall_score(true_labels, pred_labels)
f1 = f1_score(true_labels, pred_labels)
plt.figure()
plt.bar(['Precision','Recall','F1'], [pr, rc, f1])
plt.title('NER Performance Metrics')
plt.ylim(0,1)
plt.savefig('ner_metrics.png', bbox_inches='tight')
plt.show()

# 11) Plot SPU error breakdown
plt.figure()
plt.bar(['Critical','Non-critical'], [crit, noncrit])
plt.title('SPU Error Breakdown')
plt.savefig('spu_breakdown.png', bbox_inches='tight')
plt.show()

# 12) Plot MUC-5 token counts
plt.figure()
plt.bar(['COR','INC','MIS','SPU'], [COR, INC, MIS, SPU])
plt.title('MUC-5 Token Count Breakdown')
plt.ylabel('Count')
plt.savefig('muc5_counts.png', bbox_inches='tight')
plt.show()
