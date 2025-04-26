import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import pandas as pd

# --------------------------------------------------
# Positional Encoding
# --------------------------------------------------

def get_angles(pos, i, d_model):
    return pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))

def positional_encoding(max_pos, d_model):
    angle_rads = get_angles(
        np.arange(max_pos)[:, np.newaxis],
        np.arange(d_model)[np.newaxis, :],
        d_model
    )
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    return tf.cast(angle_rads[np.newaxis, ...], tf.float32)

# --------------------------------------------------
# Causal Mask
# --------------------------------------------------

def create_look_ahead_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

# --------------------------------------------------
# Feed-Forward
# --------------------------------------------------

def feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model),
    ])

# --------------------------------------------------
# Decoder Layer
# --------------------------------------------------

class DecoderLayer(layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = layers.MultiHeadAttention(num_heads, key_dim=d_model)
        self.ffn = feed_forward_network(d_model, dff)
        self.norm1 = layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = layers.Dropout(dropout)
        self.drop2 = layers.Dropout(dropout)

    def call(self, x, *, training, mask):
        attn = self.mha(query=x, value=x, key=x, attention_mask=mask)
        attn = self.drop1(attn, training=training)
        out1 = self.norm1(x + attn)

        ffn = self.ffn(out1)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(out1 + ffn)

# --------------------------------------------------
# Decoder Stack
# --------------------------------------------------

class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.dropout = layers.Dropout(dropout)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ]

    def call(self, x, *, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        for layer in self.dec_layers:
            x = layer(x, training=training, mask=mask)
        return x  # (batch, seq_len, d_model)

# --------------------------------------------------
# Full Transformer
# --------------------------------------------------

class TransformerModel(tf.keras.Model):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.decoder = Decoder(
            num_layers, d_model, num_heads, dff,
            vocab_size, max_seq_len, dropout
        )
        self.final_layer = layers.Dense(vocab_size)

    def call(self, x, *, training):
        mask = create_look_ahead_mask(tf.shape(x)[1])
        dec = self.decoder(x, training=training, mask=mask)
        return self.final_layer(dec)  # (batch, seq_len, vocab_size)

# --------------------------------------------------
# Loss & Optimizer
# --------------------------------------------------

loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none'
)
optimizer = tf.keras.optimizers.Adam(1e-4)

def loss_function(real, pred):
    mask = tf.cast(tf.not_equal(real, 0), tf.float32)
    loss_ = loss_obj(real, pred)
    loss_ *= mask
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

@tf.function
def train_step(model, inp, tar):
    with tf.GradientTape() as tape:
        logits = model(inp, training=True)
        loss = loss_function(tar, logits)
    grads = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return loss

# --------------------------------------------------
# Main
# --------------------------------------------------

def main():
    # 1) Load & tokenize
    df = pd.read_csv('train.csv')
    lyrics = df['lyric'].astype(str).tolist()

    tokenizer = tf.keras.preprocessing.text.Tokenizer(
        oov_token='[UNK]', filters=''
    )
    tokenizer.fit_on_texts(lyrics)
    # save tokenizer so load_model.py can pick it up
    with open("tokenizer.json", "w", encoding="utf-8") as f:
        f.write(tokenizer.to_json())

    seqs = tokenizer.texts_to_sequences(lyrics)

    # 2) Pad to max_seq_len+1, then split into input/target
    max_seq_len = 128
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        seqs, maxlen=max_seq_len+1, padding='post'
    )
    inputs = padded[:, :-1]   # (N, max_seq_len)
    targets = padded[:, 1:]   # (N, max_seq_len)

    # 3) Build Dataset
    batch_size = 32
    dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
    dataset = dataset.shuffle(10_000).batch(batch_size)

    # 4) Model instantiation
    vocab_size = len(tokenizer.word_index) + 1
    model = TransformerModel(
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=1024,
        vocab_size=vocab_size,
        max_seq_len=max_seq_len,
        dropout=0.1
    )

    # 5) Training loop
    epochs = 2
    for epoch in range(epochs):
        total_loss = 0.0
        batches = 0
        for inp, tar in dataset:
            loss = train_step(model, inp, tar)
            total_loss += loss
            batches += 1
        print(f"Epoch {epoch+1:2d} Loss: {total_loss / batches:.4f}")

    # 6) Save weights
    model.save_weights('lyric_transformer.weights.h5')
    print("Training complete, weights saved to lyric_transformer.weights.h5")

if __name__ == '__main__':
    main()
