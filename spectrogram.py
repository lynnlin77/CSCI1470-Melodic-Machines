import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np

def create_spectrogram(audio_file_path, output_image_path=None):
    """
    create a spectrogram from an audio file and save it as an image.
    """
    # from https://gist.github.com/TheOneAndAnu/50542ff167cabc36f78c3c416e55d5c5#file-waveplot-py
    # load audio file
    y, sr = librosa.load(audio_file_path)
    
    # short-time Fourier transform
    ft = librosa.stft(y)
    
    # convert the STFT to decibels 
    decibels = librosa.amplitude_to_db(np.abs(ft), ref=np.max)
    
    # create the spectrogram plot
    plt.figure(figsize=(10, 6))
    librosa.display.specshow(decibels, sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    
    # save the plot
    if output_image_path:
        plt.savefig(output_image_path, bbox_inches='tight', pad_inches=0.1)
    
    # return the filtered image data and the visualized plot
    return decibels, plt.gcf()

if __name__ == "__main__":
    input_audio = "test_music.mp3"
    output_image = "spectrogram.png"
    
    spectrogram, figure = create_spectrogram(input_audio, output_image)
    plt.show()