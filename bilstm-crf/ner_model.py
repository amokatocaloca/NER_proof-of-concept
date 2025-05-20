import tensorflow as tf
from tensorflow.keras import layers, optimizers, Model, Input, regularizers
import keras
from crf_layer import CRF, masked_accuracy, MaskedF1Score

def build_and_train_bilstm_crf(
    X_train, y_train,
    X_val,   y_val,    # new arguments
    X_test,  y_test,
    max_len,
    vocab_size,
    num_tags,
    embedding_matrix=None,
    embedding_dim=300,
    lstm_units=64,
    lstm_dropout=0.2,
    batch_size=32,
    epochs=50
):
    # 1) Input & Embedding
    inp = Input(shape=(max_len,), name="token_ids")
    emb = layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        weights=[embedding_matrix] if embedding_matrix is not None else None,
        trainable=(embedding_matrix is None),
        mask_zero=True,
        name="embed"
    )(inp)

    # 2) BiLSTM encoder
    bi = layers.Bidirectional(
        layers.LSTM(
            lstm_units,
            return_sequences=True,
            recurrent_dropout=lstm_dropout,
            kernel_regularizer=regularizers.l2(1e-2)
        ),
        name="bilstm"
    )(emb)

    # 3) Emission scores for CRF
    emissions = layers.TimeDistributed(
        layers.Dense(num_tags),
        name="emissions"
    )(bi)

    # 4) CRF layer
    crf = CRF(name="crf")
    outputs = crf(emissions)

    model = Model(inp, outputs, name="bilstm_crf")

    @keras.saving.register_keras_serializable()
    def crf_loss(y_true, y_pred):
        return crf.compute_loss(y_true, y_pred)

    model.compile(
        optimizer=optimizers.Adam(1e-3, global_clipnorm=1.0),
        loss=crf_loss,
        metrics=[masked_accuracy, MaskedF1Score(num_tags)]
    )

    # 5) Early stopping on validation F1
    early = tf.keras.callbacks.EarlyStopping(
        monitor="val_masked_f1", mode="max",
        patience=10, restore_best_weights=True
    )

    # 6) Fit with explicit validation_data
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        batch_size=batch_size,
        epochs=epochs,
        callbacks=[early],
        verbose=1,
        shuffle=False
    )

    # 7) Final evaluation on test set
    test_metrics = model.evaluate(X_test, y_test, verbose=1)
    return model, history, test_metrics