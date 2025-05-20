import nltk
import pandas as pd
import json
import logging

nltk.download('punkt')
logging.basicConfig(level=logging.INFO)

def offsets_to_bio(doc_text, spans):
    """
    Convert offset-based spans to BIO tags using a span-based tokenizer,
    but force any 'MISC' spans to become 'O'.
    """
    tokenizer = nltk.WhitespaceTokenizer()
    token_spans = list(tokenizer.span_tokenize(doc_text))
    tokens = [doc_text[s:e] for s, e in token_spans]

    # sort spans
    sorted_spans = sorted(spans, key=lambda s: s['start'])

    def find_label_for_token(tok_start, tok_end):
        for sp in sorted_spans:
            span_start, span_end = sp['start'], sp['end']
            raw_lbl = sp['labels'][0] if isinstance(sp['labels'], list) else sp['labels']
            # skip any 'MISC'
            if raw_lbl == "MISC":
                continue
            if not (tok_end <= span_start or tok_start >= span_end):
                return raw_lbl
        return None

    bio_tags = []
    prev_lbl = None
    for (t_start, t_end), token in zip(token_spans, tokens):
        lbl = find_label_for_token(t_start, t_end)
        if lbl is None:
            bio_tags.append((token, "O"))
            prev_lbl = None
        else:
            tag = f"I-{lbl}" if lbl == prev_lbl else f"B-{lbl}"
            bio_tags.append((token, tag))
            prev_lbl = lbl

    return bio_tags


def convert_csv_to_bio(csv_path, output_csv=None):
    """
    Convert a CSV with columns 'text' and 'label' (JSON spans)
    into a token-level BIO CSV, collapsing any MISC spans to O.
    """
    try:
        df = pd.read_csv(csv_path)
    except Exception as e:
        logging.error(f"Failed to read CSV file at {csv_path}: {e}")
        return None

    bio_rows = []
    for idx, row in df.iterrows():
        if 'text' not in row or 'label' not in row:
            logging.warning(f"Skipping row {idx}: missing 'text' or 'label'")
            continue

        text = str(row['text'])
        lbl = str(row['label']).strip()
        if not lbl:
            logging.info(f"Row {idx} has empty label")
            continue

        try:
            spans = json.loads(lbl)
        except json.JSONDecodeError as e:
            logging.warning(f"Row {idx} invalid JSON: {e}")
            continue

        bio_toks = offsets_to_bio(text, spans)
        doc_id = row.get('id', f"doc_{idx}")
        for tok, bio in bio_toks:
            bio_rows.append({
                'doc_id': doc_id,
                'token': tok,
                'bio_label': bio
            })

    out_df = pd.DataFrame(bio_rows)
    if output_csv:
        try:
            out_df.to_csv(output_csv, index=False)
            logging.info(f"Wrote BIO to {output_csv}")
        except Exception as e:
            logging.error(f"Failed to write BIO CSV: {e}")
    return out_df


if __name__ == "__main__":
    bio_df = convert_csv_to_bio("project-2-at-2025-03-23-04-43-7c53b5c2.csv",
                                "bio_output.csv")
    if bio_df is not None:
        print(bio_df.head())
