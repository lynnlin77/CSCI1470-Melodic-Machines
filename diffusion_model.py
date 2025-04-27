import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.utils import register_keras_serializable  # for serialization support
import argparse
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
from preprocess import load_encoders, load_batch, batch_generator
import json  # for persisting loss history

# os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # Set to the GPU you want to use
# Set memory growth for GPU to avoid OOM errors
gpus = tf.config.list_physical_devices("GPU")
print("Physical GPUs:", gpus)
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], True)

@register_keras_serializable()
class ConditionalUNet(Model):
    def __init__(self, freq_bins, time_frames, hidden_dim, num_artists, num_genres, **kwargs):
        # Ensure parent Model __init__ handles trainable, dtype, etc.
        super().__init__(**kwargs)
        # Save initialization args for serialization
        self.freq_bins = freq_bins
        self.time_frames = time_frames
        self.hidden_dim = hidden_dim
        self.num_artists = num_artists
        self.num_genres = num_genres

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
        self.crop1 = layers.Cropping2D(((1, 0), (0, 0)))
        self.actu1 = layers.LeakyReLU(alpha=0.2)
        self.up2  = layers.Conv2DTranspose(hidden_dim,   4, strides=2, padding='same')
        self.crop2 = layers.Cropping2D(((1, 0), (1, 0)))
        self.actu2 = layers.LeakyReLU(alpha=0.2)

        # Output layer to predict noise
        self.outc = layers.Conv2D(1, 3, padding='same')

    def call(self, x, t, artist_id, genre_id):
        t_emb = self.time_embed(tf.expand_dims(tf.cast(t, tf.float32), -1))
        a_emb = self.artist_embed(artist_id)
        g_emb = self.genre_embed(genre_id)
        cond = t_emb + a_emb + g_emb
        cond = tf.reshape(cond, [-1, 1, 1, cond.shape[-1]])
        cond = self.cond_proj(cond)

        x1 = self.act1(self.conv1(x))
        x2 = self.act2(self.conv2(x1))
        x3 = self.act3(self.conv3(x2))

        h = self.actm1(self.mid1(x3))
        h = self.actm2(self.mid2(h))
        h += cond

        h = self.up1(h)
        h = self.crop1(h)
        h = self.actu1(h) + x2
        h = self.up2(h)
        h = self.crop2(h)
        h = self.actu2(h) + x1

        return self.outc(h)

    def get_config(self):
        # Return config for serialization
        config = super().get_config()
        config.update({
            'freq_bins': self.freq_bins,
            'time_frames': self.time_frames,
            'hidden_dim': self.hidden_dim,
            'num_artists': self.num_artists,
            'num_genres': self.num_genres
        })
        return config

    @classmethod
    def from_config(cls, config):
        # Reconstruct from config
        return cls(
            config.pop('freq_bins'),
            config.pop('time_frames'),
            config.pop('hidden_dim'),
            config.pop('num_artists'),
            config.pop('num_genres'),
            **config
        )

class SpectrogramDiffusion:
    def __init__(self, model, T=100, beta_start=1e-4, beta_end=2e-2):
        self.model = model
        self.T = T
        self.betas = tf.linspace(beta_start, beta_end, T)
        self.alphas = 1.0 - self.betas
        self.acum = tf.math.cumprod(self.alphas, axis=0)
        self.sqrt_acum = tf.sqrt(self.acum)
        self.sqrt_1m_acum = tf.sqrt(1 - self.acum)

    def add_noise(self, x0, t):
        batch = tf.shape(x0)[0]
        eps = tf.random.normal(tf.shape(x0))
        a_t = tf.gather(self.sqrt_acum, t)
        am_t = tf.gather(self.sqrt_1m_acum, t)
        a_t = tf.reshape(a_t, [batch, 1, 1, 1])
        am_t = tf.reshape(am_t, [batch, 1, 1, 1])
        x_t = a_t * x0 + am_t * eps
        return x_t, eps

    def sample(self, batch, artist_id, genre_id, shape):
        x = tf.random.normal([batch, *shape])
        for i in reversed(range(self.T)):
            t = tf.fill([batch], i)
            t_n = tf.cast(t, tf.float32) / tf.cast(self.T, tf.float32)
            eps_pred = self.model(x, t_n, artist_id, genre_id)
            eps_pred = tf.clip_by_value(eps_pred, -10, 10)
            a = self.alphas[i]
            ac = self.acum[i]
            beta = self.betas[i]
            coef1 = 1.0 / tf.sqrt(a)
            coef2 = (1 - a) / tf.sqrt(1 - ac)
            noise = tf.random.normal(tf.shape(x)) if i > 0 else 0.0
            x = coef1 * (x - coef2 * eps_pred) + tf.sqrt(beta) * noise
        return (x + 1) * 0.5

@tf.function
def train_step(diffuser, x0, artist_id, genre_id, optimizer):
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
    B = tf.shape(x0)[0]
    t = tf.random.uniform([B], 0, diffuser.T, dtype=tf.int32)
    x_t, eps = diffuser.add_noise(x0, t)
    t_n = tf.cast(t, tf.float32) / tf.cast(diffuser.T, tf.float32)
    eps_pred = diffuser.model(x_t, t_n, artist_id, genre_id)
    return tf.reduce_mean(tf.square(eps - eps_pred))


def main():
    parser = argparse.ArgumentParser(description="Diffusion Model for Spectrograms")
    parser.add_argument('--pickle_dir', required=True)
    parser.add_argument('--train_list', required=True)
    parser.add_argument('--test_list', required=True)
    parser.add_argument('--artist_pkl', default='artist_encoder.pkl')
    parser.add_argument('--genre_pkl', default='genre_encoder.pkl')
    parser.add_argument('--output_dir', default='generated_samples')
    parser.add_argument('--checkpoint_dir', default='checkpoints')
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--T_steps', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--norm', choices=['minmax','zscore'], default='minmax')
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    args = parser.parse_args()

    # Load encoders & sample dimensions
    artist_le, genre_le = load_encoders(args.artist_pkl, args.genre_pkl)
    train_ids = pickle.load(open(args.train_list, 'rb'))
    test_ids  = pickle.load(open(args.test_list,  'rb'))
    x0_sample, _, _ = load_batch(train_ids[:args.batch_size], args.pickle_dir, artist_le, genre_le,
                                  batch_size=args.batch_size, norm_method=args.norm)
    _, freq_bins, time_frames, _ = x0_sample.shape

    # Build model, diffuser, optimizer
    unet = ConditionalUNet(freq_bins, time_frames, args.hidden_dim,
                            len(artist_le.classes_), len(genre_le.classes_))
    diffuser = SpectrogramDiffusion(unet, T=args.T_steps)
    optimizer = tf.keras.optimizers.Adam(args.learning_rate)

    # Prepare checkpointing (only model & epoch)
    checkpoint_epoch = tf.Variable(0, dtype=tf.int64)
    dummy_x = tf.zeros((1, freq_bins, time_frames, 1), dtype=tf.float32)
    dummy_t = tf.zeros((1,), dtype=tf.int32)
    dummy_a = tf.zeros((1,), dtype=tf.int32)
    dummy_g = tf.zeros((1,), dtype=tf.int32)
    _ = unet(dummy_x, tf.cast(dummy_t, tf.float32), dummy_a, dummy_g)

    checkpoint = tf.train.Checkpoint(model=unet, epoch=checkpoint_epoch)
    manager    = tf.train.CheckpointManager(checkpoint, args.checkpoint_dir, max_to_keep=3)

    # Restore latest checkpoint if exists
    if manager.latest_checkpoint:
        checkpoint.restore(manager.latest_checkpoint).expect_partial()
        print(f"Restored model from {manager.latest_checkpoint}, starting at epoch {int(checkpoint.epoch)}")
    else:
        print("No checkpoint found, starting from scratch.")

    # Load or initialize loss history
    loss_history_path = os.path.join(args.output_dir, 'loss_history.json')
    if os.path.exists(loss_history_path):
        with open(loss_history_path, 'r') as f:
            hist = json.load(f)
        train_losses = hist.get('train_losses', [])  # existing train losses
        val_losses   = hist.get('val_losses', [])    # existing validation losses
        print(f"Loaded loss history for {len(train_losses)} epochs")
    else:
        train_losses, val_losses = [], []
        print("Initializing new loss history")

    # Training loop
    start = int(checkpoint.epoch)
    for epoch in range(start, args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        total, batches = 0, 0
        # Training
        i = 0
        for batch in batch_generator(np.random.permutation(train_ids), args.batch_size):
            x0, a_id, g_id = load_batch(batch, args.pickle_dir, artist_le, genre_le,
                                        batch_size=args.batch_size, norm_method=args.norm)
            loss = train_step(diffuser, x0, a_id, g_id, optimizer)
            total += loss; batches += 1
            i += 1
            if i % 10 == 0:  # Print every 10 batches
                print(f"Batch {i}/{len(train_ids)//args.batch_size} - Loss: {loss:.4f}", end='\r')
        print()  # New line after batch loss
        train_loss = total / batches
        train_losses.append(float(train_loss))  # append new train loss
        print(f"Train Loss: {train_loss:.4f}")
    print("Training complete.")

        # Validation
        total, batches = 0, 0
        for batch in batch_generator(test_ids, args.batch_size):
            x0, a_id, g_id = load_batch(batch, args.pickle_dir, artist_le, genre_le,
                                        batch_size=args.batch_size, norm_method=args.norm)
            loss = test_step(diffuser, x0, a_id, g_id)
            total += loss; batches += 1
        val_loss = total / batches
        val_losses.append(float(val_loss))  # append new validation loss
        print(f"Val   Loss: {val_loss:.4f}")

        # Persist updated loss history to file
        with open(loss_history_path, 'w') as f:
            json.dump({
                'train_losses': train_losses,
                'val_losses':   val_losses
            }, f)

        # Save checkpoint
        checkpoint.epoch.assign_add(1)
        path = manager.save()
        print(f"Checkpoint saved: {path}")

    # Save final model with .keras extension
    os.makedirs(args.output_dir, exist_ok=True)
    model_path = os.path.join(args.output_dir, 'diffusion_saved.keras')
    unet.save(model_path)  # save in Keras native format

    # Plot loss curves including full history
    epochs = range(1, len(train_losses) + 1)
    plt.figure()
    plt.plot(epochs, train_losses, label='Train Loss')  # full history
    plt.plot(epochs, val_losses,   label='Val   Loss')  # full history
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')
    plt.savefig(os.path.join(args.output_dir, 'loss_curve.png'))
    plt.close()
    print(f"Saved loss curve to {os.path.join(args.output_dir, 'loss_curve.png')}")

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