# load_model.py
import tensorflow as tf, json, sys
from transformer import ConditionalTransformer  # <-- import your class

# 1) load tokenizer
tok = tf.keras.preprocessing.text.tokenizer_from_json(open("tokenizer.json").read())

# 2) load artist/genre→id maps
artist2id = json.load(open("artist2id.json"))
genre2id  = json.load(open("genre2id.json"))

# 3) rebuild model & load weights
model = ConditionalTransformer(
    num_layers=4, d_model=256, num_heads=8, dff=1024,
    vocab_size=len(tok.word_index)+1, max_seq_len=128,
    num_artists=len(artist2id), num_genres=len(genre2id)
)
model.build((None, 128))
model.load_weights("cond_transformer.weights.h5")

# 4) inference helper
def generate(artist, genre, seed, length=100, temp=1.0):
    aid = artist2id.get(artist.lower())
    gid = genre2id.get(genre.lower())
    if aid is None or gid is None:
        raise ValueError(f"Unknown artist/genre: {artist}/{genre}")
    seq = tok.texts_to_sequences([seed])[0]
    for _ in range(length):
        pad = tf.keras.preprocessing.sequence.pad_sequences(
            [seq], maxlen=128, padding='post'
        )
        logits = model(pad, artist_id=tf.constant([aid]), genre_id=tf.constant([gid]), training=False)
        logits = logits[:, len(seq), :] / temp
        nxt = int(tf.random.categorical(logits, 1)[0,0])
        seq.append(nxt)
    return " ".join(tok.index_word.get(i, "[UNK]") for i in seq)

if __name__=="__main__":
    artist, genre, seed = sys.argv[1], sys.argv[2], sys.argv[3]
    print(generate(artist, genre, seed))
