import os
import argparse
import pandas as pd
import numpy as np
import librosa
import pickle
from sklearn.preprocessing import LabelEncoder

# python preprocessing.py ^
#   --metadata_csv "./FMA metadata/metadata.csv" ^
#   --checksum "../fma_small/checksums" ^
#   --audio_dir "../fma_small/" ^
#   --output_dir "../tracks_data/" ^
#   --duration 10 ^
#   --log_scale

def parse_args():
    """Parse command-line arguments for audio preprocessing."""
    parser = argparse.ArgumentParser(
        description='Preprocess audio tracks to spectrogram pickles.'
    )
    parser.add_argument('--metadata_csv', type=str, required=True, help='Metadata CSV path')
    parser.add_argument('--checksum', type=str, required=True, help='Checksum file path')
    parser.add_argument('--audio_dir', type=str, required=True, help='Audio base directory')
    parser.add_argument('--output_dir', type=str, required=True, help='Output directory')
    parser.add_argument('--duration', type=float, default=10.0, help='Seconds to load')
    parser.add_argument('--sr', type=int, default=22050, help='Sampling rate')
    parser.add_argument('--n_fft', type=int, default=1024, help='FFT size')
    parser.add_argument('--hop_length', type=int, default=512, help='Hop length')
    parser.add_argument('--log_scale', action='store_true', help='Use dB scale')
    return parser.parse_args()


def preprocess_text(text):
    """Basic text cleaning for strings."""
    # if not isinstance(text, str):
    #     return ''
    txt = text.lower().strip()
    return ''.join(ch for ch in txt if ch.isalnum() or ch.isspace())


def load_checksum(checksum_path):
    """Build mapping from track_id to audio file path."""
    mapping = {}
    with open(checksum_path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            path = parts[1]
            tid = os.path.splitext(os.path.basename(path))[0]
            mapping[tid] = path
    return mapping


def process_track(row, mapping, args):
    """Load audio, compute spectrogram, and save pickle."""
    tid = str(row['track_id'])
    # Check if the track_id from metadata is in the mapping from checksums
    if tid not in mapping:
        print(f"Warning: {tid} not in checksum.")
        return
    # Check if the audio file exists
    if not os.path.exists(os.path.join(args.audio_dir, mapping[tid])):
        return
    # Load audio file
    y, _ = librosa.load(
        os.path.join(args.audio_dir, mapping[tid]), sr=args.sr,
        duration=args.duration
    )
    S = np.abs(librosa.stft(y, n_fft=args.n_fft, hop_length=args.hop_length))
    if args.log_scale:
        S = librosa.amplitude_to_db(S, ref=np.max)

    os.makedirs(args.output_dir, exist_ok=True)
    out = os.path.join(args.output_dir, f"{tid}_spectrogram.pkl")
    data = {
        'track_id': tid,
        'spectrogram': S,
        'metadata': {
            'track_title': row['track_title_clean'],
            'artist_name': row['artist_name_clean'],
            'genre': row['track_genre_top'],
            'genre_id': int(row['genre_id'])
        }
    }
    with open(out, 'wb') as f:
        pickle.dump(data, f)
    print(f"Saved {tid} shape={S.shape} to {out}")


def preprocess_metadata(csv_path):
    """
    Load and clean metadata, then encode genre labels.

    Returns:
        pd.DataFrame: with cleaned text and 'genre_id'.
    """
    df = pd.read_csv(csv_path)
    df = df.dropna(subset=['track_id', 'track_genre_top', 'artist_name'])
    df['track_id'] = df['track_id'].astype(int).astype(str).str.zfill(6)
    df['track_title_clean'] = df['track_title'].apply(preprocess_text)
    df['artist_name_clean'] = df['artist_name'].apply(preprocess_text)
    le = LabelEncoder()
    df['genre_id'] = le.fit_transform(df['track_genre_top'].astype(str))
    return df

def main():
    """Main entry: parse args and start preprocessing."""
    args = parse_args()
    df = preprocess_metadata(args.metadata_csv)
    mapping = load_checksum(args.checksum)
    for _, row in df.iterrows():
        process_track(row, mapping, args)

if __name__ == '__main__':
    main()
