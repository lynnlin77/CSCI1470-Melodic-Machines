Melodic Machines: A Dual-Model Approach to Artist-Conditioned Music Generation

Melodic Machines is a two-stage generative system that creates artist-conditioned music by combining transformer-based lyric generation with stable diffusion-based audio synthesis. Given an artist and genre, our model outputs both stylistically consistent lyrics and audio that reflect the artist’s sonic identity.

Our transformer generates lyrics in the stylistic voice of a given artist and genre, while our stable diffusion model synthesizes corresponding spectrograms to mimic that artist’s sound. The audio is reconstructed from spectrograms, enabling a full-text-to-sound generative pipeline.

We trained the models using the FMA audio dataset and a refined lyrics dataset sourced from Kaggle. Due to dataset incompatibilities, the audio and lyrics models were trained on different artist sets. We performed significant preprocessing—cleaning, filtering, and converting raw data into usable formats such as spectrogram matrices—to enable conditioning on genre and artist.

The transformer follows a decoder-only architecture with artist/genre embedding. The stable diffusion model uses a simplified U-Net that denoises spectrograms conditioned on time step, genre, and artist embedding.

Despite computational limitations, both models achieved promising results. The diffusion model captured brightness and stylistic patterns in spectrograms; the transformer produced lexically rich, artist-aligned lyrics. Our results were evaluated based on qualitative similarity, training loss, and stylistic consistency.

Challenges included mismatched datasets, limited GPU memory, and noise in manual datasets. We addressed these by simplifying our architectures, manually refining entries, and conditioning models on artist and genre embeddings.

If extended, we would increase model complexity, dataset scale, and improve conditioning mechanisms (e.g., cross-attention or encoder-decoder frameworks). We’d also experiment with neural vocoders and joint phase/magnitude prediction to improve audio realism.

Melodic Machines demonstrates the feasibility and creativity of a dual-model approach to multimodal generation. It deepened our understanding of conditional modeling, audio synthesis, and the ethical implications of AI-generated creative content. This project showed us the technical and philosophical considerations of AI in the arts—and how exciting this space can be.
