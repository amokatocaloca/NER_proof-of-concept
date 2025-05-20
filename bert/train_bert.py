import numpy as np
import pandas as pd
import joblib
import torch
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from transformers import (
    BertTokenizerFast,
    BertForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback
)
from seqeval.metrics import f1_score, classification_report
from transformers import set_seed
set_seed(42)

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and prepare data
df = pd.read_csv("bert/data/bio_output.csv")

# Clean and validate data
df = df[df['token'].notna()]
df['token'] = df['token'].str.strip()
df = df[df['token'] != '']

# Group documents with token lists
doc_groups = df.groupby('doc_id').agg({
    'token': list,
    'bio_label': list
}).reset_index()

df = doc_groups.rename(columns={
    'token': 'tokens',
    'bio_label': 'bio_labels'
})



# Encode labels
le = LabelEncoder()
all_labels_merged = [label for sublist in df['bio_labels'] for label in sublist]
le.fit(all_labels_merged)
joblib.dump(le, 'bert/label_encoder.joblib') 

df['encoded_labels'] = df['bio_labels'].apply(lambda x: le.transform(x))

# Stratified split
train_df, temp_df = train_test_split(df, test_size=0.3, stratify=df['bio_labels'].apply(lambda x: x[0]), random_state=42)
val_df, test_df = train_test_split(temp_df, test_size=0.5, stratify=temp_df['bio_labels'].apply(lambda x: x[0]), random_state=42)

# Initialize tokenizer
tokenizer = BertTokenizerFast.from_pretrained('bert-base-multilingual-cased', add_prefix_space=True)

class NERDataset(torch.utils.data.Dataset):
    def __init__(self, token_lists, label_lists, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.samples = []
        
        for tokens, labels in zip(token_lists, label_lists):
            if len(tokens) != len(labels) or len(tokens) == 0:
                continue
                
            encoding = tokenizer(
                tokens,
                is_split_into_words=True,
                truncation=True,
                max_length=max_length,
                padding=False,
                return_offsets_mapping=True
            )
            
            if len(encoding['input_ids']) == 0:
                continue

            word_ids = encoding.word_ids()
            label_ids = []
            previous_word_idx = None
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    try:
                        label_ids.append(labels[word_idx])
                    except IndexError:
                        label_ids.append(-100)
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx

            self.samples.append({
                'input_ids': encoding['input_ids'],
                'attention_mask': encoding['attention_mask'],
                'labels': label_ids
            })

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        return {k: torch.tensor(v) for k, v in self.samples[idx].items()}

# Prepare datasets
train_dataset = NERDataset(train_df['tokens'], train_df['encoded_labels'], tokenizer)
val_dataset = NERDataset(val_df['tokens'], val_df['encoded_labels'], tokenizer)
test_dataset = NERDataset(test_df['tokens'], test_df['encoded_labels'], tokenizer)

# Model initialization
model = BertForTokenClassification.from_pretrained(
    'bert-base-multilingual-cased',
    num_labels=len(le.classes_),
    id2label={i: label for i, label in enumerate(le.classes_)},
    label2id={label: i for i, label in enumerate(le.classes_)}
).to(device)


# Training setup
data_collator = DataCollatorForTokenClassification(tokenizer)
training_args = TrainingArguments(
    output_dir='./bert/results',
    num_train_epochs=5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    learning_rate=2e-5,
    weight_decay=0.01,
    logging_strategy="epoch",
    logging_steps=5,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    seed=42,
    metric_for_best_model='f1',
    greater_is_better=True,
    report_to="tensorboard" 
)


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)
    
    true_predictions = []
    true_labels = []
    
    for pred, lab in zip(predictions, labels):
        valid_preds = []
        valid_labs = []
        for p, l in zip(pred, lab):
            if l != -100:
                valid_preds.append(le.classes_[p])
                valid_labs.append(le.classes_[l])
        true_predictions.append(valid_preds)
        true_labels.append(valid_labs)
    
    return {'f1': f1_score(true_labels, true_predictions)}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
)

# Execute training
print("Starting training...")
trainer.train()

val_results = trainer.predict(val_dataset)

val_preds = np.argmax(val_results.predictions, axis=2)
val_labels = val_results.label_ids

true_labels_val, pred_labels_val = [], []
for pred_seq, lab_seq in zip(val_preds, val_labels):
    t_seq, p_seq = [], []
    for p, l in zip(pred_seq, lab_seq):
        if l != -100:
            t_seq.append(le.classes_[l])
            p_seq.append(le.classes_[p])
    true_labels_val.append(t_seq)
    pred_labels_val.append(p_seq)

# Validation evaluation
print("\n=== Validation Set Evaluation ===")
print("Validation metrics (HuggingFace):", val_results.metrics)

print(f"F1 Score: {f1_score(true_labels_val, pred_labels_val):.4f}")
val_metrics = trainer.evaluate(eval_dataset=val_dataset)
print(val_metrics)
print(classification_report(true_labels_val, pred_labels_val))

#  Test evaluation
print("\n=== Test Set Evaluation ===")
test_results = trainer.predict(test_dataset)
print(test_results.metrics)

# Detailed seqeval report on test
predictions = np.argmax(test_results.predictions, axis=2)
labels      = test_results.label_ids

true_labels, pred_labels = [], []
for pred_seq, lab_seq in zip(predictions, labels):
    t_seq, p_seq = [], []
    for p, l in zip(pred_seq, lab_seq):
        if l != -100:
            t_seq.append(le.classes_[l])
            p_seq.append(le.classes_[p])
    true_labels.append(t_seq)
    pred_labels.append(p_seq)

print("\n=== Test Classification Report ===")
print(classification_report(true_labels, pred_labels))
print(f"F1 Score: {f1_score(true_labels, pred_labels):.4f}")