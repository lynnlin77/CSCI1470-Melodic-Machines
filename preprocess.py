import os
import glob
import pickle
import argparse
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split


def build_encoders(artist_names, genre_names):
    """
    Fit LabelEncoders for given artist and genre names.
    Args:
        artist_names: List of artist names.
        genre_names: List of genre names.
    Returns: (artist_le, genre_le)
    """
    artist_le = LabelEncoder().fit(artist_names)
    genre_le = LabelEncoder().fit(genre_names)
    return artist_le, genre_le


def save_encoders(artist_le, genre_le, artist_pkl='artist_encoder.pkl', genre_pkl='genre_encoder.pkl'):
    """
    Save LabelEncoders to disk.
    """
    with open(artist_pkl, 'wb') as f:
        pickle.dump(artist_le, f)
    with open(genre_pkl, 'wb') as f:
        pickle.dump(genre_le, f)


def load_encoders(artist_pkl='artist_encoder.pkl', genre_pkl='genre_encoder.pkl'):
    """
    Load saved LabelEncoders from disk.
    """
    with open(artist_pkl, 'rb') as f:
        artist_le = pickle.load(f)
    with open(genre_pkl, 'rb') as f:
        genre_le = pickle.load(f)
    return artist_le, genre_le

def train_test_split_artists(pickle_dir, train_list, test_list, 
                            artist_pkl='artist_encoder.pkl', genre_pkl='genre_encoder.pkl',
                            test_size=0.2, random_state=42):
    """
    Drop artists with only one track, build encoders, then split tracks into train/test.
    Save track IDs as pickle files and encoders to disk.
    """
    # Map artists to their track IDs and collect genres
    artist_to_ids = {}
    artist_names = []
    genre_names = []
    for path in glob.glob(os.path.join(pickle_dir, '*_spectrogram.pkl')):
        tid = os.path.splitext(os.path.basename(path))[0].split('_')[0]
        with open(path, 'rb') as f:
            data = pickle.load(f)
        md = data['metadata']
        artist = md['artist_name']
        genre = md['genre']
        artist_to_ids.setdefault(artist, []).append(tid)
        artist_names.append(artist)
        genre_names.append(genre)

    # Filter to artists with >= 2 tracks
    multi_artists = [art for art, tids in artist_to_ids.items() if len(tids) > 1]

    # Gather multi-sample track IDs, artist names, and genres
    multi_ids = []
    multi_artist_names = []
    multi_genre_names = []
    for art in multi_artists:
        for tid in artist_to_ids[art]:
            multi_ids.append(tid)
            # Find the corresponding artist and genre for this track
            idx = [i for i, tid_ in enumerate(artist_to_ids[art]) if tid_ == tid][0]
            multi_artist_names.append(art)
            # Re-load genre for this track to ensure alignment
            path = os.path.join(pickle_dir, f"{tid}_spectrogram.pkl")
            with open(path, 'rb') as f:
                genre = pickle.load(f)['metadata']['genre']
            multi_genre_names.append(genre)

    # Build encoders with multi-track artists only
    artist_le, genre_le = build_encoders(multi_artist_names, multi_genre_names)
    save_encoders(artist_le, genre_le, artist_pkl, genre_pkl)

    # Encode artist labels for stratified split
    y = artist_le.transform(multi_artist_names)

    # Stratified split on multi-sample tracks
    train_ids, test_ids = train_test_split(
        multi_ids, test_size=test_size, random_state=random_state, stratify=y
    )

    # Save pickle files
    with open(train_list, 'wb') as f:
        pickle.dump(train_ids, f)
    with open(test_list, 'wb') as f:
        pickle.dump(test_ids, f)


def preprocess_spectrogram(spec, method='minmax'):
    """Normalize a spectrogram matrix."""
    if np.any(np.isnan(spec)) or np.all(spec == 0):
        raise ValueError("Spectrogram contains NaNs or is all zeros.")
    if method == 'minmax':
        vmin, vmax = spec.min(), spec.max()
        if vmax == vmin:
            return np.zeros_like(spec)  # Handle constant spectrogram
        return (spec - vmin) / (vmax - vmin)
    elif method == 'zscore':
        mu, sigma = spec.mean(), spec.std()
        if sigma < 1e-6:
            return np.zeros_like(spec)  # Handle near-constant spectrogram
        return (spec - mu) / sigma
    else:
        raise ValueError(f"Unknown normalization: {method}")


def encode_conditions(artist_names, genre_names, artist_le, genre_le):
    """Convert lists of names to integer ID arrays."""
    artist_ids = artist_le.transform(artist_names)
    genre_ids = genre_le.transform(genre_names)
    return artist_ids, genre_ids


def load_batch(track_ids, pickle_dir, artist_le, genre_le, batch_size=16, norm_method='minmax'):
    """Load and preprocess one batch; return arrays ready for model."""
    specs, artists, genres = [], [], []
    for tid in track_ids[:batch_size]:
        path = os.path.join(pickle_dir, f"{tid}_spectrogram.pkl")
        with open(path, 'rb') as f:
            d = pickle.load(f)
        spec = preprocess_spectrogram(d['spectrogram'], method=norm_method)
        specs.append(spec[..., np.newaxis])  # Add channel
        md = d['metadata']
        artists.append(md['artist_name'])
        genres.append(md['genre'])
    spec_batch = np.stack(specs, axis=0)
    a_ids, g_ids = encode_conditions(artists, genres, artist_le, genre_le)
    return spec_batch, a_ids, g_ids


def batch_generator(id_list, batch_size):
    """Yield successive batches of ids."""
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
    parser.add_argument('--norm', choices=['minmax', 'zscore'], default='minmax')
    args = parser.parse_args()

    # Split dataset and build/save encoders
    train_test_split_artists(
        args.pickle_dir, args.train_list, args.test_list,
        args.artist_pkl, args.genre_pkl
    )

    # Example: load first batch from train set
    # Load encoders
    artist_le, genre_le = load_encoders(args.artist_pkl, args.genre_pkl)
    train_ids = pickle.load(open(args.train_list, 'rb'))
    for batch_id in batch_generator(train_ids, args.batch_size):
        spec_b, art_b, gen_b = load_batch(
            batch_id, args.pickle_dir,
            artist_le, genre_le,
            batch_size=args.batch_size,
            norm_method=args.norm
        )
        print(f"Example batch: specs={spec_b.shape}, artists={art_b.shape}, genres={gen_b.shape}")
        # Break after first batch for demonstration
        break
if __name__ == '__main__':
    main()


# python preprocess.py ^
#   --pickle_dir ../tracks_data/ ^
#   --train_list train.pkl ^
#   --test_list test.pkl 