import os
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

def create_spectrogram(audio_file_path, output_image_path=None, fig_size=(10, 6), dpi=100):
    """
    create a spectrogram from an audio file and save it as an image
    """
    y, sr = librosa.load(audio_file_path)
    
    # Short-time Fourier transform
    ft = librosa.stft(y)
    
    # Convert the STFT to decibels 
    decibels = librosa.amplitude_to_db(np.abs(ft), ref=np.max)
    
    # Create the spectrogram plot
    plt.figure(figsize=fig_size)
    librosa.display.specshow(decibels, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # Save the plot
    if output_image_path:
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0, dpi=dpi)
        plt.close()
    
    return decibels

def process_audio_dataset(input_dir, output_dir, csv_path):
    """
    process all audio files and save spectrograms in a flat folder structure
    """
    os.makedirs(output_dir, exist_ok=True)
    metadata = []

    for root, dirs, files in tqdm(os.walk(input_dir), desc="Processing files"):
        artist = os.path.basename(root)
        
        if root == input_dir:
            continue 

        for file in tqdm(files, desc=f"Processing {artist}", leave=False):
            if file.lower().endswith(('.mp3', '.wav', '.flac', '.ogg', '.m4a')):
                try:
                    audio_path = os.path.join(root, file)
                    base_name = os.path.splitext(file)[0]
                    
                    safe_artist = artist.replace('/', '_').replace('\\', '_')
                    spectrogram_filename = f"{safe_artist}_{base_name}.png"
                    spectrogram_path = os.path.join(output_dir, spectrogram_filename)

                    create_spectrogram(audio_path, spectrogram_path)

                    metadata.append({
                        'artist': artist,
                        'audio_path': audio_path,
                        'spectrogram_path': spectrogram_path,
                        'filename': file
                    })

                except Exception as e:
                    print(f"Error processing {file}: {str(e)}")

    # save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(csv_path, index=False)
    print(f"Metadata saved to {csv_path}")
    print(f"Saved {len(df)} spectrograms to {output_dir}")
    return df

if __name__ == "__main__":
    # adjust paths as needed (I have a cs1470 file on my desktop)
    desktop = os.path.join(os.path.expanduser('~'), 'Desktop')
    INPUT_DIR = os.path.join(desktop, 'cs1470', 'fma_small')
    OUTPUT_DIR = os.path.join(desktop, 'cs1470', 'fma_spectrograms_flat')
    CSV_PATH = os.path.join(desktop, 'cs1470', 'fma_metadata_flat.csv')

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    process_audio_dataset(INPUT_DIR, OUTPUT_DIR, CSV_PATH)