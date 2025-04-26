import os
import pickle
import argparse
import json
import numpy as np
import tensorflow as tf
from preprocess import load_encoders, load_global_norm
from diffusion_model import SpectrogramDiffusion
import ast


def generate_samples_for_ids(model_dir,
                             pickle_dir,
                             ids,
                             output_dir,
                             T_steps=100,
                             artist_pkl='artist_encoder.pkl',
                             genre_pkl='genre_encoder.pkl',
                             norm_json='norm_params.json'):
    """
    Load a SavedModel and generate spectrogram samples for a batch of track IDs in one go.

    Args:
        model_dir (str): Directory of the SavedModel (exported via unet.save).
        pickle_dir (str): Directory containing input spectrogram pickles.
        ids (list of str): List of track IDs to generate.
        output_dir (str): Where to write one combined pickle of generated samples.
        T_steps (int): Number of diffusion timesteps.
        artist_pkl, genre_pkl (str): Paths to encoder pickles.
        norm_json (str): Path to global normalization JSON.
    """
    # Load encoders and normalization params
    artist_le, genre_le = load_encoders(artist_pkl, genre_pkl)
    vmin, vmax = load_global_norm(norm_json)

    # Determine spectrogram dims
    sample_path = os.path.join(pickle_dir, f"{ids[0]}_spectrogram.pkl")
    with open(sample_path, 'rb') as f:
        S0 = pickle.load(f)['spectrogram']
    freq_bins, time_frames = S0.shape
    shape = (freq_bins, time_frames, 1)

    # Load SavedModel (includes structure + weights)
    unet = tf.keras.models.load_model(model_dir)
    diffuser = SpectrogramDiffusion(unet, T=T_steps)

    # Prepare batch condition IDs
    artist_names = []
    genre_names = []
    for tid in ids:
        with open(os.path.join(pickle_dir, f"{tid}_spectrogram.pkl"), 'rb') as f:
            md = pickle.load(f)['metadata']
        artist_names.append(md['artist_name'])
        genre_names.append(md['genre'])
    artist_ids = artist_le.transform(artist_names)
    genre_ids  = genre_le.transform(genre_names)
    # Convert to tensors
    artist_ids_t = tf.constant(artist_ids, dtype=tf.int32)
    genre_ids_t  = tf.constant(genre_ids,  dtype=tf.int32)

    batch_size = len(ids)
    # Run diffusion sampling for the full batch
    spec_norm = diffuser.sample(
        batch=batch_size,
        artist_id=artist_ids_t,
        genre_id=genre_ids_t,
        shape=shape
    )  # shape: [B, F, T, 1]
    spec_norm = spec_norm.numpy().squeeze(-1)  # [B, F, T]

    # Inverse normalization (min-max)
    spec_rec = spec_norm * (vmax - vmin + 1e-6) + vmin  # [B, F, T]

    # Save a single pickle containing all generated samples
    os.makedirs(output_dir, exist_ok=True)
    out_path = os.path.join(output_dir, 'generated_batch.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump({
            'track_ids': ids,
            'spectrograms': spec_rec,
            'artist_ids': artist_ids,
            'genre_ids': genre_ids,
            'artist_names': artist_names,
            'genre_names': genre_names
        }, f)
    print(f"Saved generated batch of {batch_size} samples to {out_path}")


def main():
    parser = argparse.ArgumentParser(description='Generate samples via diffusion model')
    parser.add_argument('--model_dir',  required=True, help='Path to SavedModel directory')
    parser.add_argument('--pickle_dir', required=True, help='Directory of input spectrogram pickles')
    parser.add_argument('--test_list',  required=True, help='Python list of test IDs, e.g. "[\'000123\']"')
    parser.add_argument('--output_dir', required=True, help='Directory to save generated pickles')
    parser.add_argument('--artist_pkl', default='diffusion_data/artist_encoder.pkl')
    parser.add_argument('--genre_pkl',  default='diffusion_data/genre_encoder.pkl')
    parser.add_argument('--norm_json',  default='diffusion_data/norm_params.json')
    parser.add_argument('--T_steps',    type=int, default=100)
    args = parser.parse_args()

    # Parse test IDs
    ids = ast.literal_eval(args.test_list)

    generate_samples_for_ids(
        model_dir=args.model_dir,
        pickle_dir=args.pickle_dir,
        ids=ids,
        output_dir=args.output_dir,
        T_steps=args.T_steps,
        artist_pkl=args.artist_pkl,
        genre_pkl=args.genre_pkl,
        norm_json=args.norm_json
    )

if __name__ == '__main__':
    main()



# python pred_test.py ^
#   --model_dir diffusion_model/ ^
#   --pickle_dir ../tracks_data ^
#   --test_list "['056466']" ^
#   --output_dir diffusion_test_sample/