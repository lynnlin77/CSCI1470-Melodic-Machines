import tensorflow as tf
import numpy as np
import json
import pandas as pd
import os

def get_angles(pos, i, d_model):
    return pos / np.power(10000, (2 * (i//2)) / np.float32(d_model))

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
        tf.keras.layers.Dense(dff, activation='relu'),
        tf.keras.layers.Dense(d_model),
    ])

class DecoderLayer(tf.keras.layers.Layer):
    def __init__(self, d_model, num_heads, dff, dropout=0.1):
        super().__init__()
        self.mha = tf.keras.layers.MultiHeadAttention(num_heads, key_dim=d_model)
        self.ffn = feed_forward_network(d_model, dff)
        self.norm1 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.norm2 = tf.keras.layers.LayerNormalization(epsilon=1e-6)
        self.drop1 = tf.keras.layers.Dropout(dropout)
        self.drop2 = tf.keras.layers.Dropout(dropout)

    def call(self, x, *, training, mask):
        attn = self.mha(query=x, value=x, key=x, attention_mask=mask)
        attn = self.drop1(attn, training=training)
        out1 = self.norm1(x + attn)
        ffn = self.ffn(out1)
        ffn = self.drop2(ffn, training=training)
        return self.norm2(out1 + ffn)

class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_seq_len, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.embedding = tf.keras.layers.Embedding(vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.dropout = tf.keras.layers.Dropout(dropout)
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
        return x  

class ConditionalTransformer(tf.keras.Model):
    def __init__(self,
                 num_layers, d_model, num_heads, dff,
                 vocab_size, max_seq_len,
                 num_artists, num_genres,
                 dropout=0.1):
        super().__init__()
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, max_seq_len, dropout)
        self.final_layer = tf.keras.layers.Dense(vocab_size)
        self.artist_embed = tf.keras.layers.Embedding(num_artists, d_model)
        self.genre_embed = tf.keras.layers.Embedding(num_genres, d_model)

    def call(self, x, artist_id, genre_id, *, training):
        mask = create_look_ahead_mask(tf.shape(x)[1])

        dec = self.decoder(x, training=training, mask=mask) 

        a_emb = self.artist_embed(artist_id)
        g_emb = self.genre_embed(genre_id)
        cond = tf.expand_dims(a_emb + g_emb, 1)
        dec += cond  

        return self.final_layer(dec)  
    
    
def prepare_data(csv_file='train.csv'):
    df = pd.read_csv(csv_file)
    
    artist2id = {a:i for i,a in enumerate(df['artist'].unique())}
    genre2id = {g:i for i,g in enumerate(df['genre'].unique())}
    
    df['artist_id'] = df['artist'].map(artist2id)
    df['genre_id'] = df['genre'].map(genre2id)
    
    lyrics = df['lyric'].astype(str).tolist()
    tok = tf.keras.preprocessing.text.Tokenizer(oov_token='[UNK]', filters='')
    tok.fit_on_texts(lyrics)
    
    with open("tokenizer.json", "w") as f:
        f.write(tok.to_json())
    
    pd.DataFrame({
        'artist': list(artist2id.keys()),
        'id': list(artist2id.values())
    }).to_csv('artist2id.csv', index=False)
    
    pd.DataFrame({
        'genre': list(genre2id.keys()),
        'id': list(genre2id.values())
    }).to_csv('genre2id.csv', index=False)
    
    return df, tok, artist2id, genre2id

def train_model(df, tokenizer, artist2id, genre2id, epochs=20, batch_size=32, save_path='model_weights.weights.h5'):
    sequences = tokenizer.texts_to_sequences(df['lyric'].astype(str).tolist())
    
    
    max_seq_len = 128
    padded = tf.keras.preprocessing.sequence.pad_sequences(
        sequences, maxlen=max_seq_len+1, padding='post'
    )
    
    inp = padded[:, :-1]   
    tar = padded[:, 1:]    
    aid = df['artist_id'].to_numpy()
    gid = df['genre_id'].to_numpy()
    
    
    ds = tf.data.Dataset.from_tensor_slices((inp, aid, gid, tar))
    ds = ds.shuffle(10_000).batch(batch_size)
    
    
    model = ConditionalTransformer(
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=1024,
        vocab_size=len(tokenizer.word_index)+1,
        max_seq_len=max_seq_len,
        num_artists=len(artist2id),
        num_genres=len(genre2id),
        dropout=0.1
    )
    
    
    optimizer = tf.keras.optimizers.Adam(1e-4)
    loss_obj = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')
    
    def loss_function(real, pred):
        mask = tf.cast(tf.not_equal(real, 0), tf.float32)
        loss = loss_obj(real, pred) * mask
        return tf.reduce_sum(loss) / tf.reduce_sum(mask)
    
    @tf.function
    def train_step(inp, artist_id, genre_id, tar):
        with tf.GradientTape() as tape:
            logits = model(inp, artist_id, genre_id, training=True)
            loss = loss_function(tar, logits)
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))
        return loss
    
    for epoch in range(epochs):
        total_loss = 0
        steps = 0
        
        for x_batch, a_batch, g_batch, y_batch in ds:
            batch_loss = train_step(x_batch, a_batch, g_batch, y_batch)
            total_loss += batch_loss
            steps += 1
            
            if steps % 10 == 0:
                print(f"Epoch {epoch+1}, Step {steps}, Loss: {batch_loss:.4f}")
        
        print(f"Epoch {epoch+1} completed. Average Loss: {total_loss/steps:.4f}")
    
    model.save_weights(save_path)
    print(f"Model saved to {save_path}")
    
    return model

def generate_lyrics(model, artist_id, genre_id, tokenizer, max_length=100, temperature=1.0, start_text=""):
    model_max_seq_len = 128
    
    if start_text:
        input_seq = tokenizer.texts_to_sequences([start_text])[0]
        if not input_seq:
            input_seq = [1]
    else:
        input_seq = [1]
    
    if len(input_seq) >= model_max_seq_len:
        input_seq = input_seq[:model_max_seq_len-1]
    
    input_seq = tf.expand_dims(input_seq, 0) 
    
    artist_id_tensor = tf.constant([artist_id], dtype=tf.int32)
    genre_id_tensor = tf.constant([genre_id], dtype=tf.int32)
    
    output_seq = []
    
    for i in range(max_length):
        
        if tf.shape(input_seq)[1] >= model_max_seq_len:
            
            input_seq = input_seq[:, -model_max_seq_len+1:]
        
       
        predictions = model(input_seq, artist_id_tensor, genre_id_tensor, training=False)
        
       
        predictions = predictions[:, -1, :] / temperature
        
        
        predicted_id = tf.random.categorical(predictions, num_samples=1)[0, 0].numpy()
        
       
        if predicted_id == 0:
            break
            
        
        output_seq.append(predicted_id)
        
       
        input_seq = tf.concat([input_seq, tf.expand_dims([predicted_id], 0)], axis=-1)
    
    
    index_word = {v: k for k, v in tokenizer.word_index.items()}
    lyrics = ' '.join(index_word.get(id, "[UNK]") for id in output_seq)
    
    return lyrics
def load_model_and_resources():
    
    with open("tokenizer.json", "r") as f:
        tokenizer_json = json.load(f)
    tokenizer = tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(tokenizer_json))
    
    
    artist_df = pd.read_csv('artist2id.csv')
    genre_df = pd.read_csv('genre2id.csv')
    
    artist2id = dict(zip(artist_df['artist'], artist_df['id']))
    genre2id = dict(zip(genre_df['genre'], genre_df['id']))
    
    
    model = ConditionalTransformer(
        num_layers=4,
        d_model=256,
        num_heads=8,
        dff=1024,
        vocab_size=len(tokenizer.word_index) + 1,
        max_seq_len=128,
        num_artists=len(artist2id),
        num_genres=len(genre2id),
        dropout=0.1
    )
    
    
    model.load_weights('model_weights.h5')
    
    return model, tokenizer, artist2id, genre2id

def main():
    
    if not os.path.exists('model_weights.h5'):
        print("Training new model...")
        df, tokenizer, artist2id, genre2id = prepare_data()
        model = train_model(df, tokenizer, artist2id, genre2id)
    else:
        print("Loading existing model...")
        model, tokenizer, artist2id, genre2id = load_model_and_resources()
    
    
    print("\nAvailable Artists:")
    for artist in artist2id.keys():
        print(f"- {artist}")
    
    print("\nAvailable Genres:")
    for genre in genre2id.keys():
        print(f"- {genre}")
    
    
    artist_name = input("\nEnter artist name (e.g., 'dua lipa'): ")
    genre_name = input("Enter genre (e.g., 'pop'): ")
    
    
    if artist_name not in artist2id:
        print(f"Artist '{artist_name}' not found in training data. Using default.")
        artist_name = list(artist2id.keys())[0]
    
    if genre_name not in genre2id:
        print(f"Genre '{genre_name}' not found in training data. Using default.")
        genre_name = list(genre2id.keys())[0]
    
    
    start_text = input("Enter optional starting text (or press Enter to skip): ")
    
    
    try:
        temp = 1.0
    except ValueError:
        temp = 1.0
    
    
    print(f"\nGenerating lyrics in the style of {artist_name} ({genre_name})...\n")
    lyrics = generate_lyrics(
        model,
        artist2id[artist_name],
        genre2id[genre_name],
        tokenizer,
        max_length=150,
        temperature=temp,
        start_text=start_text
    )
    
    formatted_lyrics = format_lyrics(lyrics)
    
    print("=" * 50)
    print(formatted_lyrics)
    print("=" * 50)

def format_lyrics(lyrics):
    lyrics = ' '.join(lyrics.split())
    
    for marker in ['pre', 'refrain', 'breakdown', 'post']:
        lyrics = lyrics.replace(f" {marker} ", f"\n\n[{marker.upper()}]\n")
    
    lines = []
    current_line = []
    word_count = 0
    
    for word in lyrics.split():
        current_line.append(word)
        word_count += 1
        
        if word.endswith(('.', '?', '!', ',')) and word_count > 4 or word_count > 8:
            lines.append(' '.join(current_line))
            current_line = []
            word_count = 0
    
    if current_line:
        lines.append(' '.join(current_line))
    
    return '\n'.join(lines)

if __name__ == "__main__":
    main()
