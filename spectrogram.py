import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm
import taglib

def create_spectrogram(audio_file_path, output_image_path=None, fig_size=(10, 6), dpi=100):
    y, sr = librosa.load(audio_file_path, sr=None)
    
    # Short-time Fourier transform
    ft = librosa.stft(y)
    
    # Convert the STFT to decibels 
    decibels = librosa.amplitude_to_db(np.abs(ft), ref=np.max)
    
    # Create and save the spectrogram plot
    plt.figure(figsize=fig_size)
    librosa.display.specshow(decibels, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    if output_image_path:
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
    plt.close()
    
    return decibels

def load_fma_metadata(metadata_path):
    tracks = pd.read_csv(metadata_path)
    return pd.Series(tracks.artist.values, index=tracks.track_id).to_dict()

def get_track_id_from_filename(filename):
    base_name = os.path.splitext(os.path.basename(filename))[0]
    try:
        return int(base_name)
    except ValueError:
        return base_name

def extract_artist_from_file(audio_path):
    audio_file = taglib.File(audio_path)
    if 'ARTIST' in audio_file.tags and audio_file.tags['ARTIST']:
        return audio_file.tags['ARTIST'][0]
    return "Unknown Artist"

def process_audio_dataset(input_dir, output_dir, csv_path, metadata_path=None):
    os.makedirs(output_dir, exist_ok=True)
    
    track_to_artist = {}
    if metadata_path and os.path.exists(metadata_path):
        track_to_artist = load_fma_metadata(metadata_path)
        print(f"Loaded metadata for {len(track_to_artist)} tracks")
    
    metadata = []
    skipped_files = []
    
    for root, _, files in tqdm(os.walk(input_dir), desc="Processing files"):
        for file in files:
            if not file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                continue
                
            audio_path = os.path.join(root, file)
            track_id = get_track_id_from_filename(file)
            
            artist = track_to_artist.get(track_id, extract_artist_from_file(audio_path))
            artist_output_dir = os.path.join(output_dir, artist.replace('/', '_'))
            os.makedirs(artist_output_dir, exist_ok=True)
            
            base_name = os.path.splitext(file)[0]
            spectrogram_path = os.path.join(artist_output_dir, f"{base_name}.png")
            
            try:
                decibels = create_spectrogram(audio_path, spectrogram_path)
                if decibels is not None:
                    metadata.append({
                        'track_id': track_id,
                        'artist': artist,
                        'audio_path': audio_path,
                        'spectrogram_path': spectrogram_path
                    })
                else:
                    skipped_files.append(audio_path)
            except Exception as e:
                print(f"Error processing {file}: {str(e)}")
                skipped_files.append(audio_path)
    
    if metadata:
        pd.DataFrame(metadata).to_csv(csv_path, index=False)
        print(f"Metadata saved to {csv_path}")
    
    if skipped_files:
        with open(os.path.join(os.path.dirname(csv_path), "skipped_files.txt"), "w") as f:
            f.write("\n".join(skipped_files))
        print(f"List of {len(skipped_files)} skipped files saved")

if __name__ == "__main__":
    # Need to change the path (I have a file called "cs1470" on my desktop)
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    INPUT_DIR = os.path.join(desktop, 'cs1470', 'fma_small')
    OUTPUT_DIR = os.path.join(desktop, 'cs1470', 'fma_spectrograms')
    CSV_PATH = os.path.join(desktop, 'cs1470', 'fma_metadata.csv')
    METADATA_PATH = os.path.join(desktop, 'cs1470', 'fma_metadata', 'tracks.csv')
    
    process_audio_dataset(INPUT_DIR, OUTPUT_DIR, CSV_PATH, METADATA_PATH)