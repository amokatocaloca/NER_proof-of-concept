
## Prerequisites

- Install the required Python packages by running:

```bash
pip install -r requirements.txt
```

## Project Structure

BERT
**train_bert.py**  
  Loads the BIO-tagged CSV, builds and fine-tunes a multilingual BERT token-classification head on the stratified train/val splits, and saves out the best checkpoint.

- **app.py**  
 Runs the inference to test specific model output

- **evaluate.py**  
  Executes the full evaluation suite:  
  1. Inference on the validation and test splits (producing seqeval classification reports).  
  2. Inference over the entire dataset (“model performance” report).  
  3. MUC-5/SPU error analysis with breakdown counts.  

- **data/**  
  Contains `bio_output.csv` (raw BIO-tagged token/label pairs)  
  
- **results/**  
  Stores the results of BERT model fine-tuning

BiLSTM-CRF

- **vocab.py**  
  Reads the BIO-tagged CSV (`bio_output.csv`), builds word & tag vocabularies, aligns bilingual FastText embeddings, and splits into train/val/test arrays saved as `.npz` files.

- **train.py**  
  Loads the preprocessed splits and embedding matrix, constructs the BiLSTM+CRF model via `build_and_train_bilstm_crf`, trains with early stopping, and writes out the best checkpoint (`ner_model.keras`).

- **crf_layer.py**  
  Defines the custom `CRF` Keras layer along with its log-likelihood loss and masked metrics (`masked_accuracy`, `MaskedF1Score`) used during training and evaluation.

- **ner_model.py**  
  Implements the `build_and_train_bilstm_crf` function—wiring together the Embedding, BiLSTM, TimeDistributed Dense, and CRF layers, compiling with the custom loss/metrics, and running model.fit.

- **data_utils.py**  
  Helper routines for loading and cleaning data

- **app.py**  
   Runs the inference to test specific model output

- **bio_output.csv**  
  The raw token-level annotated dataset (with `doc_id`, `token`, `bio_label`) used to generate vocabularies, embeddings, and train/val/test splits.


## Building and Running the Model Training Script

FOR BERT

1. **Run the script**  
   Make sure that the data files are in place and all dependencies are installed
   Open a terminal in the project directory and run:
   ```bash
    python bert/train_bert.py
   ```
   This command runs the training.

2. **Evaluation**  
   ```bash
    python bert/evaluate.py
   ```

3. **Running Inference**  
To test how the models handles data, run
   ```bash
    streamlit run bert/app.py
   ```

FOR BiLSTM-CRF


1. **Build the vocabulary**  
```bash
python bilstm-crf/vocab.py \
--input_csv bilstm-crf/bio_output.csv \
--word_vocab_file bilstm-crf/word_vocab.npy \
--tag_vocab_file bilstm-crf/tag_vocab.npy \
--trimmed_npz_file bilstm-crf/embeddings.npz \
--split_data_file bilstm-crf/split_data.npz
```
2. **Run training**  
```bash
    python bilstm-crf/trian.py
```
3. **Evaluate**  
```bash
    python bilstm-crf/evaluate.py
```
4. **Run Inference**  
```bash
    streamlit run bilstm-crf/app.py
```
## Running the software pipeline

```bash
    streamlit run main.py
```
# NER_proof-of-concept
