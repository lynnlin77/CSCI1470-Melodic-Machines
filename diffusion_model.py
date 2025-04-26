import tensorflow as tf
from tensorflow.keras import layers, Model

class ConditionalUNet(Model):
    def __init__(self, freq_bins, time_frames, hidden_dim, num_artists, num_genres):
        super().__init__()
        # time step embedding
        self.time_embed = tf.keras.Sequential([
            layers.Dense(hidden_dim),
            layers.LeakyReLU(alpha=0.2),
            layers.Dense(hidden_dim),
            layers.LeakyReLU(alpha=0.2),
        ])
        # artist + genre embedding
        self.artist_embed = layers.Embedding(num_artists, hidden_dim)
        self.genre_embed  = layers.Embedding(num_genres,  hidden_dim)

        # encoder architecture
        self.conv1 = layers.Conv2D(hidden_dim,   3, padding='same')
        self.act1  = layers.LeakyReLU(alpha=0.2)
        self.conv2 = layers.Conv2D(hidden_dim*2, 4, strides=2, padding='same')
        self.act2  = layers.LeakyReLU(alpha=0.2)
        self.conv3 = layers.Conv2D(hidden_dim*4, 4, strides=2, padding='same')
        self.act3  = layers.LeakyReLU(alpha=0.2)

        # middle blocks
        self.mid1 = layers.Conv2D(hidden_dim*4, 3, padding='same')
        self.actm1 = layers.LeakyReLU(alpha=0.2)
        self.mid2 = layers.Conv2D(hidden_dim*4, 3, padding='same')
        self.actm2 = layers.LeakyReLU(alpha=0.2)

        # decoder architecture
        self.up1  = layers.Conv2DTranspose(hidden_dim*2, 4, strides=2, padding='same')
        self.actu1 = layers.LeakyReLU(alpha=0.2)
        self.up2  = layers.Conv2DTranspose(hidden_dim,   4, strides=2, padding='same')
        self.actu2 = layers.LeakyReLU(alpha=0.2)

        # output - predict noise
        self.outc = layers.Conv2D(1, 3, padding='same')

    def call(self, x, t, artist_id, genre_id):
        # embed conditioning
        t_emb = self.time_embed(tf.expand_dims(tf.cast(t, tf.float32), -1))  
        a_emb = self.artist_embed(artist_id)                                 
        g_emb = self.genre_embed(genre_id)                                    
        cond  = t_emb + a_emb + g_emb                                         
        cond  = tf.reshape(cond, [-1,1,1,cond.shape[-1]])                     

        # encoder
        x1 = self.act1(self.conv1(x))
        x2 = self.act2(self.conv2(x1))
        x3 = self.act3(self.conv3(x2))

        # middle + cond
        h  = self.actm1(self.mid1(x3))
        h  = self.actm2(self.mid2(h))
        h += cond

        # decoder
        h  = self.actu1(self.up1(h)) + x2
        h  = self.actu2(self.up2(h)) + x1

        # noise prediction
        return self.outc(h)

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
        """
        Forward diffusion: x_t = √αₜ x₀ + √(1−αₜ) ε
        Returns noisy spectrogram x_t and the added noise ε.
        """
        batch = tf.shape(x0)[0]
        eps   = tf.random.normal(tf.shape(x0))
        a_t   = tf.gather(self.sqrt_acum, t)
        am_t  = tf.gather(self.sqrt_1m_acum, t)
        a_t   = tf.reshape(a_t, [batch,1,1,1])
        am_t  = tf.reshape(am_t,[batch,1,1,1])
        x_t = a_t * x0 + am_t * eps
        return x_t, eps

    def sample(self, batch, artist_id, genre_id, shape):
        """
        Reverse diffusion (sampling) starting from pure noise.
        """
        x = tf.random.normal([batch, *shape])
        for i in reversed(range(self.T)):
            t    = tf.fill([batch], i)
            t_n  = tf.cast(t, tf.float32) / tf.cast(self.T, tf.float32)
            eps_pred = self.model(x, t_n, artist_id, genre_id)
            α     = self.alphas[i]
            αc    = self.acum[i]
            β     = self.betas[i]
            coef1 = 1.0 / tf.sqrt(α)
            coef2 = (1 - α) / tf.sqrt(1 - αc)
            noise = tf.random.normal(tf.shape(x)) if i > 0 else 0.0
            x = coef1 * (x - coef2 * eps_pred) + tf.sqrt(β) * noise
        return (x + 1) * 0.5

@tf.function
def train_step(diffuser, x0, artist_id, genre_id, optimizer):
    """
    One training step: sample t, add noise, predict noise, backprop MSE loss.
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




def main():
    freq_bins, time_frames = 64, 64 
    hidden_dim  = 64
    T_steps     = 100 
    num_artists = 50               
    num_genres  = 20
    batch_size  = 16

    unet     = ConditionalUNet(freq_bins, time_frames, hidden_dim, num_artists, num_genres)
    diffuser = SpectrogramDiffusion(unet, T=T_steps)

    optimizer = tf.keras.optimizers.Adam(1e-4)

    for epoch in range(10):
        x0         = tf.random.uniform([batch_size, freq_bins, time_frames, 1], -1, 1)
        artist_id  = tf.random.uniform([batch_size], 0, num_artists, dtype=tf.int32)
        genre_id   = tf.random.uniform([batch_size], 0, num_genres,  dtype=tf.int32)
        loss = train_step(diffuser, x0, artist_id, genre_id, optimizer)
        tf.print("Epoch", epoch, "Loss", loss)

    samples = diffuser.sample(batch_size, artist_id, genre_id, (freq_bins, time_frames, 1))
    print("Generated spectrograms shape:", samples.shape)

if __name__ == "__main__":
    main()
