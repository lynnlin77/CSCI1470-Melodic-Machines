import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# Hardcoded spectrogram target dimensions
TARGET_FREQ_BINS = 513
TARGET_TIME_FRAMES = 431
TARGET_SHAPE = (TARGET_FREQ_BINS, TARGET_TIME_FRAMES)


def build_encoders(artist_names, genre_names):
    """
    Fit LabelEncoders for given artist and genre names.
    Returns: (artist_le, genre_le)
    """
    artist_le = LabelEncoder().fit(artist_names)
    genre_le  = LabelEncoder().fit(genre_names)
    return artist_le, genre_le


def save_encoders(artist_le, genre_le,
                  artist_pkl='artist_encoder.pkl', genre_pkl='genre_encoder.pkl'):
    """Save LabelEncoders to disk."""
    with open(artist_pkl, 'wb') as f:
        pickle.dump(artist_le, f)
    with open(genre_pkl, 'wb') as f:
        pickle.dump(genre_le, f)


def load_encoders(artist_pkl='artist_encoder.pkl', genre_pkl='genre_encoder.pkl'):
    """Load saved LabelEncoders from disk."""
    with open(artist_pkl, 'rb') as f:
        artist_le = pickle.load(f)
    with open(genre_pkl, 'rb') as f:
        genre_le  = pickle.load(f)
    return artist_le, genre_le


def train_test_split_artists(pickle_dir, train_list, test_list,
                             artist_pkl, genre_pkl,
                             test_size=0.2, random_state=42):
    """
    Drop artists with only one track, then split remaining tracks into train/test.
    Save track IDs lists and encoders.
    """
    # Map artists to their track IDs and collect conditions
    artist_to_ids = {}
    artist_names = []
    genre_names = []
    for path in glob.glob(os.path.join(pickle_dir, '*_spectrogram.pkl')):
        tid = os.path.splitext(os.path.basename(path))[0].split('_')[0]
        with open(path, 'rb') as f:
            data = pickle.load(f)
        art = data['metadata']['artist_name']
        gen = data['metadata']['genre']
        artist_to_ids.setdefault(art, []).append(tid)
        artist_names.append(art)
        genre_names.append(gen)

    # Filter to artists with >=2 tracks
    multi_artists = [art for art, tids in artist_to_ids.items() if len(tids) > 1]

    # Gather multi-sample track_ids and conditions
    multi_ids, multi_art_names, multi_gen_names = [], [], []
    for art in multi_artists:
        for tid in artist_to_ids[art]:
            multi_ids.append(tid)
            multi_art_names.append(art)
            with open(os.path.join(pickle_dir, f"{tid}_spectrogram.pkl"),'rb') as f:
                multi_gen_names.append(pickle.load(f)['metadata']['genre'])

    # Build and save encoders
    artist_le, genre_le = build_encoders(multi_art_names, multi_gen_names)
    save_encoders(artist_le, genre_le, artist_pkl, genre_pkl)

    # Encode and split
    y = artist_le.transform(multi_art_names)
    train_ids, test_ids = train_test_split(
        multi_ids, test_size=test_size,
        random_state=random_state, stratify=y
    )

    # Save ID lists
    with open(train_list, 'wb') as f:
        pickle.dump(train_ids, f)
    with open(test_list, 'wb') as f:
        pickle.dump(test_ids, f)


def preprocess_spectrogram(spec, method='minmax'):
    """Normalize a spectrogram matrix using fixed -80 to 0 dB range or z-score."""
    if method == 'minmax':
        # always normalize from -80dB to 0dB
        vmin, vmax = -80.0, 0.0
        return (spec - vmin) / (vmax - vmin)  # maps [-80,0] to [0,1]
    elif method == 'zscore':
        mu, sigma = spec.mean(), spec.std()
        return (spec - mu) / (sigma + 1e-6)
    else:
        raise ValueError(f"Unknown normalization: {method}")


def encode_conditions(artist_names, genre_names, artist_le, genre_le):
    """Convert lists of names to integer ID arrays."""
    a_ids = artist_le.transform(artist_names)
    g_ids = genre_le.transform(genre_names)
    return a_ids, g_ids


def pad_or_trim(spec, target_shape=TARGET_SHAPE):
    """Pad with zeros or truncate spectrogram to target_shape."""
    f, t = spec.shape
    th, tw = target_shape
    if f < th or t < tw:
        out = np.zeros(target_shape, dtype=spec.dtype)
        out[:f, :t] = spec
        return out
    else:
        return spec[:th, :tw]


def load_batch(track_ids, pickle_dir, artist_le, genre_le,
               batch_size=16, norm_method='minmax'):
    """Load and preprocess a batch; return numpy arrays for model."""
    specs, arts, gens = [], [], []
    for tid in track_ids[:batch_size]:
        path = os.path.join(pickle_dir, f"{tid}_spectrogram.pkl")
        with open(path, 'rb') as f:
            d = pickle.load(f)
        # Normalize
        spec = preprocess_spectrogram(d['spectrogram'], method=norm_method)
        # Pad or trim to fixed size
        spec = pad_or_trim(spec)
        # Add channel dimension
        specs.append(spec[..., np.newaxis])
        arts.append(d['metadata']['artist_name'])
        gens.append(d['metadata']['genre'])
    # Stack into batch array
    spec_batch = np.stack(specs, axis=0)
    a_ids, g_ids = encode_conditions(arts, gens, artist_le, genre_le)
    return spec_batch, a_ids, g_ids


def batch_generator(id_list, batch_size):
    """Yield successive batches of IDs."""
    for i in range(0, len(id_list), batch_size):
        yield id_list[i: i + batch_size]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--pickle_dir', required=True)
    parser.add_argument('--train_list', required=True)
    parser.add_argument('--test_list', required=True)
    parser.add_argument('--artist_pkl', default='artist_encoder.pkl')
    parser.add_argument('--genre_pkl', default='genre_encoder.pkl')
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--norm', choices=['minmax','zscore'], default='minmax')
    args = parser.parse_args()

    # Split dataset and build/save encoders
    train_test_split_artists(
        args.pickle_dir, args.train_list, args.test_list,
        args.artist_pkl, args.genre_pkl
    )

    # Example: load first batch from train set
    artist_le, genre_le = load_encoders(args.artist_pkl, args.genre_pkl)
    train_ids = pickle.load(open(args.train_list, 'rb'))
    for batch in batch_generator(train_ids, args.batch_size):
        spec_b, art_b, gen_b = load_batch(
            batch, args.pickle_dir, artist_le, genre_le,
            batch_size=args.batch_size, norm_method=args.norm
        )
        print(f"Example batch: specs={spec_b.shape}, artists={art_b.shape}, genres={gen_b.shape}")
        break

if __name__ == '__main__':
    main()


# python preprocess.py ^
#   --pickle_dir ../tracks_data/ ^
#   --train_list diffusion_data/train.pkl ^
#   --test_list diffusion_data/test.pkl 