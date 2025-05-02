import os
import pickle
import argparse
import numpy as np
import tensorflow as tf
from preprocess import load_encoders
import ast
from diffusion_model import ConditionalUNet, SpectrogramDiffusion  # import custom class

# Set memory growth for GPU to avoid OOM errors
gpus = tf.config.list_physical_devices("GPU")
print("Physical GPUs:", gpus)
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)


def generate_sample_for_id(model_dir,
                           pickle_dir,
                           track_id,
                           output_dir,
                           T_steps=200,
                           artist_pkl='diffusion_data/artist_encoder.pkl',
                           genre_pkl='diffusion_data/genre_encoder.pkl'):
    """
    Load a SavedModel and generate one spectrogram sample for a given track ID.
    Saves both the generated spectrogram and its associated metadata.
    """
    # Load encoders
    artist_le, genre_le = load_encoders(artist_pkl, genre_pkl)

    # Determine spectrogram dims and metadata
    pickle_path = os.path.join(pickle_dir, f"{track_id}_spectrogram.pkl")
    with open(pickle_path, 'rb') as f:
        data = pickle.load(f)
    S0 = data['spectrogram']
    metadata = data['metadata']
    freq_bins, time_frames = S0.shape
    shape = (freq_bins, time_frames, 1)

    # Load model
    unet = tf.keras.models.load_model(
        model_dir,
        custom_objects={'ConditionalUNet': ConditionalUNet}
    )
    diffuser = SpectrogramDiffusion(unet, T=T_steps)

    # Encode artist and genre
    artist_id = artist_le.transform([metadata['artist_name']])[0]
    genre_id  = genre_le.transform([metadata['genre']])[0]
    artist_id_t = tf.constant([artist_id], dtype=tf.int32)
    genre_id_t  = tf.constant([genre_id], dtype=tf.int32)

    # Run diffusion sampling
    spec_norm = diffuser.sample(
        batch=1,
        artist_id=artist_id_t,
        genre_id=genre_id_t,
        shape=shape
    ).numpy().squeeze(-1)  # [F, T]

    # Inverse normalization (assuming [0,1] -> [-80,0] dB)
    vmin, vmax = -80.0, 0.0
    spec_rec = spec_norm * (vmax - vmin) + vmin  # [F, T]

    # Prepare output
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, f"{track_id}_generated.pkl")
    with open(out_path, 'wb') as f:
        pickle.dump({
            'track_id': track_id,
            'spectrogram': spec_rec,
            'metadata': {
                'artist_id': int(artist_id),
                'genre_id': int(genre_id),
                'artist_name': metadata['artist_name'],
                'genre': metadata['genre'],
                'original_shape': (freq_bins, time_frames)
            }
        }, f)
    print(f"Saved generated sample and metadata for track {track_id} to {out_path}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Generate one sample via diffusion model')
    parser.add_argument('--model_dir',  required=True, help='Path to SavedModel or .keras file')
    parser.add_argument('--pickle_dir', required=True, help='Directory of input spectrogram pickles')
    parser.add_argument('--track_id',   required=True, help='Single track ID to generate')
    parser.add_argument('--output_dir', required=True, help='Directory to save generated pickle')
    parser.add_argument('--T_steps',    type=int, default=200)
    args = parser.parse_args()

    generate_sample_for_id(
        model_dir=args.model_dir,
        pickle_dir=args.pickle_dir,
        track_id=args.track_id,
        output_dir=args.output_dir,
        T_steps=args.T_steps,
        artist_pkl='diffusion_data/artist_encoder.pkl',
        genre_pkl='diffusion_data/genre_encoder.pkl'
    )


# python pred_test.py \
#   --model_dir diffusion_saved.keras \
#   --pickle_dir ../tracks_data \
#   --track_id 000197 \
#   --output_dir diffusion_data/generated_samples
  