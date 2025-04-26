import tensorflow as tf
from tensorflow.keras import layers, Model
import argparse
import pickle
import numpy as np
import os
from preprocess import load_encoders, load_batch, batch_generator

class ConditionalUNet(Model):
    def __init__(self, freq_bins, time_frames, hidden_dim, num_artists, num_genres):
        super().__init__()
        # Time step embedding layers
        self.time_embed = tf.keras.Sequential([
            layers.Dense(hidden_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(hidden_dim),
            layers.LeakyReLU(alpha=0.2),
        ])
        # Artist and genre embedding layers
        self.artist_embed = layers.Embedding(num_artists, hidden_dim)
        self.genre_embed  = layers.Embedding(num_genres,  hidden_dim)

        # Projection layer for conditioning
        self.cond_proj = layers.Dense(hidden_dim * 4)

        # Encoder architecture
        self.conv1 = layers.Conv2D(hidden_dim,   3, padding='same')
        self.act1  = layers.LeakyReLU(alpha=0.2)
        self.conv2 = layers.Conv2D(hidden_dim*2, 4, strides=2, padding='same')
        self.act2  = layers.LeakyReLU(alpha=0.2)
        self.conv3 = layers.Conv2D(hidden_dim*4, 4, strides=2, padding='same')
        self.act3  = layers.LeakyReLU(alpha=0.2)

        # Middle blocks
        self.mid1 = layers.Conv2D(hidden_dim*4, 3, padding='same')
        self.actm1 = layers.LeakyReLU(alpha=0.2)
        self.mid2 = layers.Conv2D(hidden_dim*4, 3, padding='same')
        self.actm2 = layers.LeakyReLU(alpha=0.2)

        # Decoder architecture
        self.up1  = layers.Conv2DTranspose(hidden_dim*2, 4, strides=2, padding='same')
        self.crop1 = layers.Cropping2D(((1, 0), (0, 0)))  # Crop 1 pixel from top
        self.actu1 = layers.LeakyReLU(alpha=0.2)
        self.up2  = layers.Conv2DTranspose(hidden_dim,   4, strides=2, padding='same')
        self.crop2 = layers.Cropping2D(((1, 0), (1, 0)))  # Crop 1 pixel from top and right
        self.actu2 = layers.LeakyReLU(alpha=0.2)

        # Output layer to predict noise
        self.outc = layers.Conv2D(1, 3, padding='same')

    def call(self, x, t, artist_id, genre_id):
        # Embed time step, artist, and genre conditions
        t_emb = self.time_embed(tf.expand_dims(tf.cast(t, tf.float32), -1))  
        a_emb = self.artist_embed(artist_id)                                 
        g_emb = self.genre_embed(genre_id)                                    
        cond = t_emb + a_emb + g_emb                                         
        cond = tf.reshape(cond, [-1, 1, 1, cond.shape[-1]])                 
        cond = self.cond_proj(cond)  # Project to hidden_dim*4 channels

        # Encoder
        x1 = self.act1(self.conv1(x))  # [batch, 513, 431, 64]
        x2 = self.act2(self.conv2(x1))  # [batch, 257, 216, 128]
        x3 = self.act3(self.conv3(x2))  # [batch, 129, 108, 256]

        # Middle blocks with conditioning
        h = self.actm1(self.mid1(x3))
        h = self.actm2(self.mid2(h))
        h += cond  # Broadcast condition across spatial dimensions

        # Decoder with skip connections
        h = self.up1(h)              # [batch, 258, 216, 128]
        h = self.crop1(h)            # [batch, 257, 216, 128]
        h = self.actu1(h) + x2       # [batch, 257, 216, 128]
        h = self.up2(h)              # [batch, 514, 432, 64]
        h = self.crop2(h)            # [batch, 513, 431, 64]
        h = self.actu2(h) + x1       # [batch, 513, 431, 64]

        # Predict noise
        return self.outc(h)          # [batch, 513, 431, 1]

class SpectrogramDiffusion:
    def __init__(self, model, T=100, beta_start=1e-4, beta_end=2e-2):
        self.model = model
        self.T = T
        # Define noise schedule
        self.betas = tf.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.acum = tf.math.cumprod(self.alphas, axis=0)
        self.sqrt_acum = tf.sqrt(self.acum)
        self.sqrt_1m_acum = tf.sqrt(1 - self.acum)

    def add_noise(self, x0, t):
        """
        Forward diffusion: x_t = √αₜ x₀ + √(1−αₜ) ε
        Returns noisy spectrogram x_t and the added noise ε.
        """
        batch = tf.shape(x0)[0]
        eps = tf.random.normal(tf.shape(x0))
        a_t = tf.gather(self.sqrt_acum, t)
        am_t = tf.gather(self.sqrt_1m_acum, t)
        a_t = tf.reshape(a_t, [batch, 1, 1, 1])
        am_t = tf.reshape(am_t, [batch, 1, 1, 1])
        x_t = a_t * x0 + am_t * eps
        return x_t, eps

    def sample(self, batch, artist_id, genre_id, shape):
        """
        Reverse diffusion: Generate spectrogram from noise.
        Returns normalized spectrogram in [0, 1].
        """
        x = tf.random.normal([batch, *shape])
        for i in reversed(range(self.T)):
            t = tf.fill([batch], i)
            t_n = tf.cast(t, tf.float32) / tf.cast(self.T, tf.float32)
            eps_pred = self.model(x, t_n, artist_id, genre_id)
            eps_pred = tf.clip_by_value(eps_pred, -10, 10)  # Clip predicted noise for stability
            a = self.alphas[i]
            ac = self.acum[i]
            beta = self.betas[i]
            coef1 = 1.0 / tf.sqrt(a)
            coef2 = (1 - a) / tf.sqrt(1 - ac)
            noise = tf.random.normal(tf.shape(x)) if i > 0 else 0.0
            x = coef1 * (x - coef2 * eps_pred) + tf.sqrt(beta) * noise
        spec_norm = (x + 1) * 0.5
        # Log the range of spec_norm for debugging
        print("Final spec_norm range:", tf.reduce_min(spec_norm).numpy(), tf.reduce_max(spec_norm).numpy())
        return spec_norm

@tf.function
def train_step(diffuser, x0, artist_id, genre_id, optimizer):
    """
    Perform one training step: Add noise, predict noise, compute MSE loss, and update weights.
    """
    with tf.GradientTape() as tape:
        B = tf.shape(x0)[0]
        t = tf.random.uniform([B], 0, diffuser.T, dtype=tf.int32)
        x_t, eps = diffuser.add_noise(x0, t)
        t_n = tf.cast(t, tf.float32) / tf.cast(diffuser.T, tf.float32)
        eps_pred = diffuser.model(x_t, t_n, artist_id, genre_id)
        loss = tf.reduce_mean(tf.square(eps - eps_pred))
    grads = tape.gradient(loss, diffuser.model.trainable_variables)
    optimizer.apply_gradients(zip(grads, diffuser.model.trainable_variables))
    return loss

@tf.function
def test_step(diffuser, x0, artist_id, genre_id):
    """
    Compute validation loss on test batch without updating weights.
    """
    B = tf.shape(x0)[0]
    t = tf.random.uniform([B], 0, diffuser.T, dtype=tf.int32)
    x_t, eps = diffuser.add_noise(x0, t)
    t_n = tf.cast(t, tf.float32) / tf.cast(self.T, tf.float32)
    eps_pred = diffuser.model(x_t, t_n, artist_id, genre_id)
    loss = tf.reduce_mean(tf.square(eps - eps_pred))
    return loss

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description="Diffusion Model for Spectrograms")
    parser.add_argument('--pickle_dir', required=True, help="Directory containing spectrogram pickle files")
    parser.add_argument('--train_list', required=True, help="Path to train track IDs pickle")
    parser.add_argument('--test_list', required=True, help="Path to test track IDs pickle")
    parser.add_argument('--artist_pkl', default='artist_encoder.pkl', help="Path to artist encoder pickle")
    parser.add_argument('--genre_pkl', default='genre_encoder.pkl', help="Path to genre encoder pickle")
    parser.add_argument('--output_dir', default='generated_samples', help="Directory to save final model")
    parser.add_argument('--checkpoint_dir', default='checkpoints', help="Directory to save checkpoints")
    parser.add_argument('--hidden_dim', type=int, default=64, help="Hidden dimension of U-Net")
    parser.add_argument('--T_steps', type=int, default=100, help="Number of diffusion timesteps")
    parser.add_argument('--batch_size', type=int, default=32, help="Training batch size")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs")
    parser.add_argument('--norm', choices=['minmax', 'zscore'], default='minmax', help="Spectrogram normalization method")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="Learning rate for Adam optimizer")
    args = parser.parse_args()

    # Load artist and genre encoders
    artist_le, genre_le = load_encoders(args.artist_pkl, args.genre_pkl)

    # Get number of artists and genres
    num_artists = len(artist_le.classes_)
    num_genres = len(genre_le.classes_)

    # Load train and test track IDs
    train_ids = pickle.load(open(args.train_list, 'rb'))
    test_ids = pickle.load(open(args.test_list, 'rb'))

    # Load a sample batch to determine spectrogram dimensions
    sample_batch = load_batch(
        train_ids[:args.batch_size], args.pickle_dir, artist_le, genre_le,
        batch_size=args.batch_size, norm_method=args.norm
    )
    x0_sample, artist_ids, genre_ids = sample_batch
    _, freq_bins, time_frames, _ = x0_sample.shape

    # Initialize U-Net model and diffusion process
    unet = ConditionalUNet(freq_bins, time_frames, args.hidden_dim, num_artists, num_genres)
    diffuser = SpectrogramDiffusion(unet, T=args.T_steps)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    # Initialize checkpoint to save model, optimizer, and epoch
    checkpoint = tf.train.Checkpoint(
        unet=unet,
        optimizer=optimizer,
        epoch=tf.Variable(0)
    )
    # Manage checkpoints, keeping the latest 3
    checkpoint_manager = tf.train.CheckpointManager(
        checkpoint,
        directory=args.checkpoint_dir,
        max_to_keep=3
    )

    # Restore the latest checkpoint if available
    latest_checkpoint = checkpoint_manager.latest_checkpoint
    if latest_checkpoint:
        checkpoint.restore(latest_checkpoint)
        print(f"Restored checkpoint from {latest_checkpoint}, trained for {int(checkpoint.epoch)} epochs")
    else:
        print("No checkpoint found, starting training from scratch")

    # Training loop starting from restored epoch
    start_epoch = int(checkpoint.epoch)
    for epoch in range(start_epoch, args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        # Shuffle training IDs for each epoch
        shuffled_ids = np.random.permutation(train_ids)
        total_loss = 0
        num_batches = 0
        for batch_ids in batch_generator(shuffled_ids, args.batch_size):
            x0, artist_id, genre_id = load_batch(
                batch_ids, args.pickle_dir, artist_le, genre_le,
                batch_size=args.batch_size, norm_method=args.norm
            )
            loss = train_step(diffuser, x0, artist_id, genre_id, optimizer)
            total_loss += loss
            num_batches += 1
        if num_batches > 0:
            print(f"Training Loss: {total_loss / num_batches:.4f}")

        # Compute validation loss
        total_test_loss = 0
        num_test_batches = 0
        for batch_ids in batch_generator(test_ids, args.batch_size):
            x0, artist_id, genre_id = load_batch(
                batch_ids, args.pickle_dir, artist_le, genre_le,
                batch_size=args.batch_size, norm_method=args.norm
            )
            test_loss = test_step(diffuser, x0, artist_id, genre_id)
            total_test_loss += test_loss
            num_test_batches += 1
        if num_test_batches > 0:
            print(f"Validation Loss: {total_test_loss / num_test_batches:.4f}")

        # Save checkpoint after each epoch
        checkpoint.epoch.assign_add(1)
        checkpoint_path = checkpoint_manager.save()
        print(f"Saved checkpoint to {checkpoint_path}")

    # Create output directory for final model
    os.makedirs(args.output_dir, exist_ok=True)

    # Save final model weights
    path = os.path.join(args.output_dir, 'diffusion_model')
    unet.save(path, save_format='tf')
    print(f"Final model weights saved to {path}")

if __name__ == "__main__":
    main()

# python diffusion_model.py ^
#   --pickle_dir ../tracks_data/ ^
#   --train_list diffusion_data/train8.pkl ^
#   --test_list diffusion_data/test2.pkl ^
#   --artist_pkl diffusion_data/artist_encoder.pkl ^
#   --genre_pkl diffusion_data/genre_encoder.pkl ^
#   --output_dir . ^
#   --hidden_dim 64 ^
#   --T_steps 100 ^
#   --batch_size 32 ^
#   --epochs 2 ^
#   --norm minmax ^
#   --learning_rate 1e-4