import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

# using tf as main framefork for building/training model and numpy for generating positional encodings 

# add positional encoding to the input embedding so mdoel knwos the orders of word
def get_angles(pos, i, d_model):
    """
    Generates a matrix of angles used to create sine and cosine waves for each token positon/dimension

    Args:
    pos: positon in the sequence
    i: dimension index
    d_model: total embedding dimensions

    Return:
    Angle matrix of shape (position, d_model)
        - sinusodial along each dimension -- high/low freq
        - unique per position
    """
    angle_rates = pos / np.power(10000, (2 * (i // 2)) / np.float32(d_model))
    return angle_rates

def positional_encoding(pos, d_model):
    """
    Find the fixed matrix of sine/cosine that enbcode positions across the dimensions
    values are added to the token embedding so the position of each word can be knwon

    Args:
    pos: number of positoins in the sequecne
    d_model: total embedding dimensions

    Return:
    tensor of shaope (1, pos, d_model)
        - 1 for each braodcasting across batch
        - pos for each row corresponding to a position
        - d_model for vector for each psotion, sinusodial values
    """

    # turns matrix into raw angles -- note that it is yet sine/cosine
    angle_rads = get_angles(np.arange(pos)[:, np.newaxis], np.arange(d_model)[np.newaxis, :], d_model)

    # sine for even dimensions, cosine for odd dimensios
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])

    # add batch dimen at front,g etting new shape (1, pos, d_model) to be added to embedded token
    pos_encoding = angle_rads[np.newaxis, ...]
    
    # return pos encoding as tensor
    return tf.cast(pos_encoding, dtype=tf.float32)


def create_look_ahead_mask(size):
    """
    Model should not be able to see the future tokens to prevent model 
    from attending to tokens after the current one during the training process.
    Create a mask matrix to ensure that for position i, the model can only attend to positions leq i

    Args:
    size: sequence length

    Return:
    A square tensor of shape (size, size), where it is a bool mask where
        - 0 is where atten is allowed
        - 1 is where atten is not allowed (blocked off)
    """

    # mnake square matrix of 1's, keeps the lower trainge and zero out everything above it
    mask = 1 -  tf.linalg.band_part(tf.ones((size, size)), -1, 0)
    return mask


def feed_forward_network(d_model, dff):
    """
    Defines the feed-forward network whiuch is applied after self-atten to allow
    model to transform representations at each postion independently
    In other words, give each word its own mini neural network

    Args:
    d_model: size of token embedding
    dff: size of hidden layer

    Return:
    tf.keras.Sequential model
        - two dense layers whre first increases the dimension and second projects it back
    """

    # applies fully connected layer to each position, increase embeeding to dff and add nonlinearity
    # then projects the result back to the original embedding size
    return tf.keras.Sequential([
        layers.Dense(dff, activation='relu'),
        layers.Dense(d_model)
    ])

class DecoderLayer(tf.keras.layers.Layer):

    def __init__(self, d_model, num_heads, dff, dropout_rate=0.1):
        """
        Defines one Transformer decoder block. Each DecoderLayer:
            - Applies masked multi-head self-attention
            - Processes tokens independently using a feed-forward network
            - Uses residual connections and layer normalization

        Args:
        d_model: size of embedding
        num_heads: number of attention heads
        dff: hidden size in the FFN
        dropout_rate: dropout probability
        """
        super(DecoderLayer, self).__init__()
        # masked self-atten layer
        self.mha = layers.MultiHeadAttention(num_heads=num_heads, key_dim=d_model)

        # feed forward network to process each token independently
        self.ffn = feed_forward_network(d_model, dff)

        # layer norm to stabilize training, one for atten, one for FFN
        self.layernorm1 = layers.LayerNormalization(epsilon=1e-6)
        self.layernorm2 = layers.LayerNormalization(epsilon=1e-6)

        # regularization to randomly drop units -- prevents overfit
        self.dropout1 = layers.Dropout(dropout_rate)
        self.dropout2 = layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Executes the forward pass through the decoder layer.、

        Args:
        x: input tensor (batch_size, seq_len, d_model)
        training: boolean, training mode toggle for dropout
        mask: look-ahead mask for causal attention

        Returns:
        Output tensor after attention and feed-forward transformations
        """
        # compute masked self-atten => (batch_size, seq_len, d_model)
        atten_output = self.mha(query=x, key=x, value=x, attention_mask=mask)

        # add atten output to x and apply layer norm
        out1 = self.layernorm1(x + self.dropout1(atten_output, training=training))

        # ffn on norm atten and then residual connection + norm, then return
        ffn_output  =self.ffn(out1)
        out2 = self.layernorm2(out1 + self.dropout2(ffn_output, training=training))
        return out2
    

    
class Decoder(tf.keras.layers.Layer):
    def __init__(self, num_layers, d_model, num_heads, dff, target_vocab_size, max_seq_len, dropout_rate=0.1):
        """
        Decoder stack composed of multiple DecoderLayer blocks

        Args:
        num_layers: num of stacked decoder layers
        d_model: dimension of embedding and atten layers
        num_heads: num of atten heads
        dff: dimension of feed forward layer
        target_vocab_size: vocab size for output
        max_seq_len: max len of input seq
        dropout_rate: drop probability
        """
        super(Decoder, self).__init__()
        # store model params, embedding, vectors of dim, sinisodial pos encoding, stack decoder blocks, apply dropout to embedding and pos encoding
        self.d_model = d_model
        self.num_layers = num_layers
        self.embedding = layers.Embedding(target_vocab_size, d_model)
        self.pos_encoding = positional_encoding(max_seq_len, d_model)
        self.dec_layers = [DecoderLayer(d_model, num_heads, dff, dropout_rate) for i in range(num_layers)]
        self.dropout = layers.Dropout(dropout_rate)

    def call(self, x, training, mask):
        """
        Forward pass through decoder stack

        Args:
        x: input tensor containing token indices
        training: bool flag for dropout
        mask: look ahead mask to prevent the model attedning to future tokens

        Returns:
        Tesnor of shape (batch_size, seq_len, d_model) after final decoder layer
        """

        # find tokem embeddings for each ID in the batch
        seq_len = tf.shape(x)[1]
        x = self.embedding(x)

        # scale mebeddings, add pos encoding, apply dropout
        x *= tf.math.sqrt(tf.cast(self.d_model, tf.float32))
        x += self.pos_encoding[:, :seq_len, :]
        x = self.dropout(x, training=training)

        # iterate through each decoder layer and update the represenations
        for i in range(self.num_layers):
            x = self.dec_layers[i](x, training, mask)

        return x

class TransformerModel(tf.keras.Model):

    def __init__(self, num_layers, d_model, num_heads, dff, vocab_size, max_seq_len, dropout_rate=0.1):
        """
        Full transformer decoder model for lyric generation

        Args:
        num_layers: number of decoder layers
        d_model: embedding dimensions
        num_heads: num of atten heads
        dff: hidden size in feed-forward network
        vocab_size: num of tokens in output vocab
        max_seq_len: max len of input seq
        dropout_rate: dropout rate
        """
        super(TransformerModel, self).__init__()

        # decoder stack
        self.decoder = Decoder(num_layers, d_model, num_heads, dff, vocab_size, max_seq_len, dropout_rate)

        # final projection layers: d_model => vocab_size
        self.final_layer = layers.Dense(vocab_size)

    def call(self, x, training):
        """
        Foward pass of the transformer model

        Args:
        x: input tensor
        training: bool flag for training model

        Returns:
        logits => (batch_size, seq_len, vocab_size)
        """

        mask = create_look_ahead_mask(tf.shape(x)[1])
        dec_output = self.decoder(x, training, mask)
        final_output = self.final_layer(dec_output)
        return final_output
    


loss_object = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True, reduction='none')

def loss_function(real, pred):
    """
    Computes loss ignoring padding tokens
    """
    mask = tf.math.logical_not(tf.math.equal(real, 0)) 
    loss_ = loss_object(real, pred)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask  
    return tf.reduce_sum(loss_) / tf.reduce_sum(mask)

optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)


@tf.function
def train_step(model, inp, tar):
    """
    Single training step
    Args:
        model: Transformer model
        inp: input sequence (batch_size, seq_len)
        tar: target sequence (batch_size, seq_len)
    Returns:
        loss value
    """
    with tf.GradientTape() as tape:
        predictions = model(inp, training=True)
        loss = loss_function(tar, predictions)

    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


def train(model, dataset, epochs):
    """
    Trains the model
    Args:
        model: TransformerModel
        dataset: tf.data.Dataset yielding (input, target) pairs
        epochs: number of epochs to train
    """
    for epoch in range(epochs):
        total_loss = 0
        num_batches = 0

        for batch, (inp, tar) in enumerate(dataset):
            batch_loss = train_step(model, inp, tar)
            total_loss += batch_loss
            num_batches += 1

            if batch % 100 == 0:
                tf.print('Epoch', epoch+1, 'Batch', batch, 'Loss', batch_loss)

        epoch_loss = total_loss / num_batches
        print(f'Epoch {epoch+1} Loss: {epoch_loss:.4f}')

