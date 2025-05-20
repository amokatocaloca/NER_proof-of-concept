import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from gensim.models import KeyedVectors

from data_utils import load_csv_data

# ─── 1. Load aligned embeddings once ─────────────────────────────────────────
RU_EMBEDDINGS = KeyedVectors.load_word2vec_format('wiki.multi.ru.vec')
EN_EMBEDDINGS = KeyedVectors.load_word2vec_format('wiki.multi.en.vec')

def get_bilingual_embedding(word):
    """Get embedding from Russian or English aligned space, or random if OOV."""
    try:
        return RU_EMBEDDINGS[word]
    except KeyError:
        try:
            return EN_EMBEDDINGS[word]
        except KeyError:
            return np.random.normal(size=300)

def build_vocabularies(grouped):
    """Build word and tag vocabs from grouped DataFrame."""
    all_tokens = []
    for _, row in grouped.iterrows():
        toks = row["token"]
        toks = [t if (t in RU_EMBEDDINGS or t in EN_EMBEDDINGS) else "<UNK>" for t in toks]
        all_tokens.extend(toks)

    # 2) Word vocab
    word_counts = Counter(all_tokens)
    word_vocab = {"<PAD>": 0, "<UNK>": 1}
    idx = 2
    for w, _ in word_counts.most_common():
        if w not in word_vocab:
            word_vocab[w] = idx
            idx += 1

    # 3) Tag vocab
    unique_tags = {t for tags in grouped["bio_label"] for t in tags}
    unique_tags.add("<PAD>")
    sorted_tags = sorted(
        unique_tags,
        key=lambda x: (x != "<PAD>", x != "O", x)
    )
    tag_vocab = {tag: i for i, tag in enumerate(sorted_tags)}
    return word_vocab, tag_vocab

def build_bilingual_embeddings(word_vocab, dim=300):
    """Return embedding matrix shaped (V,dim)."""
    V = max(word_vocab.values()) + 1
    M = np.zeros((V, dim), dtype=np.float32)
    for w, i in word_vocab.items():
        if w == "<PAD>":
            continue
        M[i] = get_bilingual_embedding(w) if w not in ["<UNK>"] else np.random.normal(scale=0.6, size=dim)
    return M

def create_train_val_test_arrays(grouped, word_vocab, tag_vocab,
                                 test_size=0.15, random_state=42):
    # 1) Build Xpad and ypad as before…
    X, y = [], []
    pad_w = word_vocab["<PAD>"]
    pad_t = tag_vocab["<PAD>"]
    for _, row in grouped.iterrows():
        toks = row["token"]
        labs = row["bio_label"]
        X.append([word_vocab.get(t, word_vocab["<UNK>"]) for t in toks])
        y.append([tag_vocab.get(l, tag_vocab["O"]) for l in labs])
    L = max(len(s) for s in X)
    Xpad = np.array([s + [pad_w] * (L - len(s)) for s in X], dtype=np.int32)
    ypad = np.array([s + [pad_t] * (L - len(s)) for s in y], dtype=np.int32)

    # 2) Stratify key
    strat = grouped["strat_key"].replace({"HAS_OTHER": 0, "NO_OTHER": 1}).values

    # 3) First split: (train+val) vs test, keep strat_temp
    X_temp, X_test, y_temp, y_test, strat_temp, _ = train_test_split(
        Xpad, ypad, strat,
        test_size=test_size,
        stratify=strat,
        random_state=random_state
    )

    # 4) Compute relative validation size
    val_rel = test_size / (1.0 - test_size)

    # 5) Second split: train vs val, stratifying on strat_temp
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp,
        test_size=val_rel,
        stratify=strat_temp,
        random_state=random_state
    )

    return X_train, X_val, X_test, y_train, y_val, y_test

def main(args):
    # 5) Load labels
    _, grouped = load_csv_data(args.input_csv)
    
    # 6) Print original distribution
    original_labels = [l for doc in grouped["bio_label"] for l in doc]
    print("\n=== Original Label Distribution ===")
    print(Counter(original_labels).most_common(10))


    # 7) Create safe stratification
    grouped["strat_key"] = grouped["bio_label"].apply(
        lambda labels: "HAS_OTHER" if "OTHER" in labels else "NO_OTHER"
    )

    # 8) Verify stratification classes
    strat_counts = grouped["strat_key"].value_counts()
    print("\nStratification Classes:")
    print(strat_counts)
    
    # 9) Build and save vocabs
    wv, tv = build_vocabularies(grouped)
    

    # 10) Save outputs
    np.save(args.word_vocab_file, wv)
    np.save(args.tag_vocab_file, tv)
    print("\n→ Vocabularies saved")

    if args.trimmed_npz_file:
        emb = build_bilingual_embeddings(wv)
        np.savez(args.trimmed_npz_file, embeddings=emb)
        print(f"→ Embeddings saved to {args.trimmed_npz_file}")

    if args.split_data_file:
        X_train, X_val, X_test, y_train, y_val, y_test = create_train_val_test_arrays(
            grouped, wv, tv,
            test_size=args.test_size,
            random_state=args.random_state
        )
        np.savez(
            args.split_data_file,
            X_train=X_train,  y_train=y_train,
            X_val=  X_val,    y_val=  y_val,
            X_test= X_test,   y_test= y_test
        )
        print(f"→ Train/Val/Test splits saved to {args.split_data_file}")

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Prepare vocabs+embeddings+splits for NER")
    p.add_argument("--input_csv", required=True)
    p.add_argument("--word_vocab_file", required=True)
    p.add_argument("--tag_vocab_file", required=True)
    p.add_argument("--trimmed_npz_file")
    p.add_argument("--split_data_file")
    p.add_argument("--test_size", type=float, default=0.15)
    p.add_argument("--random_state", type=int, default=42)
    args = p.parse_args()
    main(args)