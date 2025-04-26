import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import json
import sys

# -----------------------------------------------
# Re‐define the exact same architecture pieces
# -----------------------------------------------

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

def create_look_ahead_mask(seq_len):
    return 1 - tf.linalg.band_part(tf.ones((seq_len, seq_len)), -1, 0)

def feed_forward_network(d_model, dff):
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model),
    ])

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

class Decoder(layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff,
                 vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.dropout = layers.Dropout(dropout)
        self.dec_layers = [
            DecoderLayer(d_model, num_heads, dff, dropout)
            for _ in range(num_layers)
        ]
        self.d_model = d_model

    def call(self, x, *, training, mask):
        seq_len = tf.shape(x)[1]
        x = self.embedding(x) * tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)
        for layer in self.dec_layers:
            x = layer(x, training=training, mask=mask)
        return x

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
        return self.final_layer(dec)

# -----------------------------------------------
# Load tokenizer
# -----------------------------------------------

with open("tokenizer.json", "r", encoding="utf-8") as f:
    tok = tf.keras.preprocessing.text.tokenizer_from_json(f.read())

# -----------------------------------------------
# Build & load the model
# -----------------------------------------------

num_layers = 4
d_model    = 256
num_heads  = 8
dff        = 1024
max_seq_len= 128
vocab_size = len(tok.word_index) + 1

model = TransformerModel(
    num_layers, d_model, num_heads, dff,
    vocab_size, max_seq_len, dropout=0.1
)

model.build(input_shape=(None, 128))
model.load_weights("lyric_transformer.weights.h5")
# -----------------------------------------------
# Inference helper
# -----------------------------------------------

def generate_lyrics(seed_text, gen_len=50, temp=1.0):
    # convert seed to sequence
    seq = tok.texts_to_sequences([seed_text])[0]
    for _ in range(gen_len):
        inp = tf.keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=max_seq_len, padding="pre"
        )
        preds = model(inp, training=False)
        logits = preds[:, -1, :] / temp
        next_id = tf.random.categorical(logits, num_samples=1)[0,0].numpy()
        seq.append(int(next_id))
    return " ".join(tok.index_word.get(i, "[UNK]") for i in seq)

# -----------------------------------------------
# Command-line entry point
# -----------------------------------------------

if __name__ == "__main__":
    seed = sys.argv[1] if len(sys.argv) > 1 else "once upon a time"
    print(generate_lyrics(seed, gen_len=100))
