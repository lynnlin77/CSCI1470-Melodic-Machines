# Melodic Machines: A Dual-Model Approach to Artist-Conditioned Music Generation

## Overview
Melodic Machines is a two-stage framework for artist-conditioned music generation that combines:
1. A transformer-based lyric generator that creates artist-style lyrics
2. A stable diffusion model that generates artist-style spectrograms (convertible to audio)

## Features
- Artist-Conditioned Generation: Input an artist name to generate content in their style
- Dual-Modality: Produces both lyrics and matching audio spectrograms
- Genre Control: Additional genre parameter for finer stylistic control
- Custom Architectures: Simplified but effective implementations tailored for music generation

## Installation
```bash
git clone https://github.com/[your-username]/melodic-machines.git
cd melodic-machines
pip install -r requirements.txt


### Lyric Generation

```python
from lyric_generator import generate_lyrics

# Generate lyrics in the style of an artist
lyrics = generate_lyrics(artist="Taylor Swift", genre="pop", max_length=200)
print(lyrics)
```

### Audio Spectrogram Generation

```python
from audio_generator import generate_spectrogram, convert_to_audio

# Generate and save spectrogram
spectrogram = generate_spectrogram(artist="Dua Lipa", genre="pop")
spectrogram.save("generated_spectrogram.png")

# Convert to audio (requires additional audio processing libraries)
audio = convert_to_audio(spectrogram)
audio.export("generated_audio.wav", format="wav")
```

---

## Datasets

We used two primary datasets:

- **FMA (Free Music Archive)**: For training the audio spectrogram diffusion model  
- **Top Artists Lyrics Dataset (Kaggle)**: For training the lyric generation model  

> Due to licensing restrictions, we cannot redistribute the original datasets, but preprocessing scripts are provided in the `data_processing/` directory.

---

##  Model Architectures

### 1. Lyric Generator
- Decoder-only Transformer
- Artist and genre conditioning via embedding layers
- Trained on curated lyric datasets

### 2. Audio Diffusion Model
- Simplified U-Net architecture
- Operates on spectrogram image matrices
- Conditioned on artist, genre, and timestep embeddings

---

## Results

Sample outputs are available in the `examples/` directory:

- Generated lyrics for various artists  
- Original vs. generated spectrogram visual comparisons  
- Audio samples converted from generated spectrograms  
