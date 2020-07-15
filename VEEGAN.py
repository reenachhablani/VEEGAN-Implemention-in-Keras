"""
## Setup  
"""

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import *
from tensorflow.keras.layers import InputLayer

import numpy as np
rows, columns,channel = 28,28,1
latent_dim = 100
SGDop = keras.optimizers.SGD(0.0003 , 0.5)
ADAMop = keras.optimizers.Adam(0.0003, 0.5)

"""
## Prepare MNIST data
"""

# We use both the training & test MNIST digits.
batch_size = 64
(x_train, _), (x_test, _) = keras.datasets.mnist.load_data()
all_digits = np.concatenate([x_train, x_test])
all_digits = all_digits.astype("float32") / 255
all_digits = np.reshape(all_digits, (-1, 28, 28, 1))
dataset = tf.data.Dataset.from_tensor_slices(all_digits)
dataset = dataset.shuffle(buffer_size=1024).batch(batch_size).prefetch(32)


"""
## Encoder
"""

class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon
        
 def encoder(kernel, filter, rows, columns, channel):
    X = keras.Input(shape=(rows, columns, channel))
    model = Conv2D(filters=filter, kernel_size=kernel, strides=1, padding='same')(X)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*2, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*4, kernel_size=kernel, strides=1, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*8, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Flatten()(model)

    mean = Dense(100)(model)
    logsigma = Dense(100, activation='tanh')(model)
    latent = Sampling()([mean, logsigma])
    meansigma = keras.Model([X], [mean, logsigma, latent])
    return meansigma

# encoder
E = encoder(5, 32, rows, columns, channel)
E.compile(optimizer=SGDop, loss='mse') 
E.summary()

"""
## Generator/Decoder
"""

def generator(kernel, filter, rows, columns, channel):
    X = keras.Input(shape=(100,))

    model = Dense(7*7*rows*columns)(X)
    model = Reshape((7,7,rows*columns))(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU()(model)

    model = Conv2DTranspose(filter*8, kernel_size=kernel, strides=1, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU()(model)
    
    model = Conv2DTranspose(filter*4, kernel_size=kernel, strides=1, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU()(model)

    model = Conv2DTranspose(filter*2, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU()(model)

    model = Conv2DTranspose(filter, kernel_size=kernel, strides=1, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU()(model)

    model = Conv2DTranspose(1, kernel_size=kernel, strides=2, padding='same')(model)
    model = Activation('tanh')(model)
    
    model = keras.Model(X, model)
    return model

# generator/decoder
G = generator(5, 32, rows, columns, channel)
G.compile(optimizer=SGDop, loss='mse')
print(G)
G.summary()

"""
## discriminator
"""

def discriminator(kernel, filter, rows, columns, channel):
    X = keras.Input(shape=(rows, columns, channel))

    model = Conv2D(filter*2, kernel_size=kernel, strides=1, padding='same')(X)
    model = LeakyReLU(alpha=0.2)(model)
    model = Dropout(0.3)(model)

    model = Conv2D(filter*4, kernel_size=kernel, strides=2, padding='same')(model)
    model = BatchNormalization(epsilon=1e-5)(model)
    model = LeakyReLU(alpha=0.2)(model)

    model = Conv2D(filters=filter*8, kernel_size=kernel, strides=1, padding='same')(model)
    
    dec = BatchNormalization(epsilon=1e-5)(model)
    dec = LeakyReLU(alpha=0.2)(dec)
    dec = Dropout(0.3)(dec)

    dec = Flatten()(dec)
    dec = Dense(1, activation='sigmoid')(dec)

    output = keras.Model(inputs = X, outputs = dec)
    return output

# discriminator
D = discriminator(5, 32, rows, columns, channel)
D.compile(optimizer=SGDop, loss='mse')
D.summary()

"""
## VAE
"""

# VAE
X = Input(shape=(rows, columns, channel))
# latent_rep = E(X)[0]
# output = G(latent_rep)
E_mean, E_logsigma, Z = E(X)
output = G(Z)
VAE = keras.Model(X, output)
reconstruction_loss = tf.reduce_mean(keras.losses.binary_crossentropy(X, output))
reconstruction_loss *= 28 * 28
kl_loss = 1 + E_logsigma - tf.square(E_mean) - tf.exp(E_logsigma)
kl_loss = tf.reduce_mean(kl_loss)
kl_loss *= -0.5
total_loss = reconstruction_loss + kl_loss
VAE.add_loss(total_loss)
VAE.compile(optimizer=ADAMop)
VAE.summary()

"""
##Train
"""


class GAN(keras.Model):
    def __init__(self, encoder, discriminator, generator, VAE, latent_dim):
        super(GAN, self).__init__()
        self.encoder = encoder
        self.discriminator = discriminator
        self.generator = generator
        self.VAE = VAE
        self.latent_dim = latent_dim

    def compile(self, d_optimizers, g_optimizer, loss_fn):
        super(GAN, self).compile()
        self.optimizers = optimizers
        self.loss_fn = loss_fn

    def train_step(self, real_images):
        if isinstance(real_images, tuple):
            real_images = real_images[0]
        # Sample random points in the latent space
        batch_size = tf.shape(real_images)[0]
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        latent_vect = E.predict(real_images)[0]
  
        # Decode them to fake images
        generated_images = self.generator(random_latent_vectors)
        encoded_images = self.generator(latent_vect)
        # Combine them with real images
        combined_images = tf.concat([generated_images, real_images], axis=0)

        # Assemble labels discriminating real from fake images
        labels = tf.concat(
            [tf.ones((batch_size, 1)), tf.zeros((batch_size, 1))], axis=0
        )
        # Add random noise to the labels - important trick!
        labels += 0.05 * tf.random.uniform(tf.shape(labels))

        # Train the discriminator
        with tf.GradientTape() as tape:
            predictions = self.discriminator(combined_images)
            d_loss = self.loss_fn(labels, predictions)
        grads = tape.gradient(d_loss, self.discriminator.trainable_weights)
        self.optimizers.apply_gradients(
            zip(grads, self.discriminator.trainable_weights)
            predictions_enc = self.discriminator(encoded_images)
            d_loss_enc = self.loss_fn(tf.ones((batch_size, 1)), predictions_enc)
        grads = tape.gradient(d_loss_enc, self.encoder.trainable_weights)
        self.optimizers.apply_gradients(
            zip(grads, self.encoder.trainable_weights)
        )

        # Sample random points in the latent space
        random_latent_vectors = tf.random.normal(shape=(batch_size, self.latent_dim))

        # Assemble labels that say "all real images"
        misleading_labels = tf.zeros((batch_size, 1))

        # Train the generator (note that we should *not* update the weights
        # of the discriminator)!
        with tf.GradientTape() as tape:
            predictions = self.discriminator(self.generator(random_latent_vectors))
            g_loss = self.loss_fn(misleading_labels, predictions)
        grads = tape.gradient(g_loss, self.generator.trainable_weights)
        self.optimizers.apply_gradients(zip(grads, self.generator.trainable_weights))
        with tf.GradientTape() as tape:
            predictions = self.VAE(real_images)
            vae_loss = self.loss_fn(None, predictions)
        grads = tape.gradient(vae_loss, self.VAE.trainable_weights)
        self.optimizers.apply_gradients(zip(grads, self.VAE.trainable_weights))
        
        return {
            "d_loss": d_loss,
            "d_loss_enc":  d_loss_enc, 
            "g_loss": g_loss_enc,
            "g_loss_gen":g_loss_gen,
            "vae_loss":vae_loss}

"""
##display
"""


class GANMonitor(keras.callbacks.Callback):
    def __init__(self, num_img=3, latent_dim=128):
        self.num_img = num_img
        self.latent_dim = latent_dim

    def on_epoch_end(self, epoch, logs=None):
        random_latent_vectors = tf.random.normal(shape=(self.num_img, self.latent_dim))
        generated_images = self.model.generator(random_latent_vectors)
        generated_images *= 255
        generated_images.numpy()
        for i in range(self.num_img):
            img = keras.preprocessing.image.array_to_img(generated_images[i])
            img.save("generated_img_{i}_{epoch}.png".format(i=i, epoch=epoch))


epochs = 30

gan = GAN(E,D,G,VAE, latent_dim=latent_dim)
gan.compile(
    optimizers=keras.optimizers.Adam(learning_rate=0.0003),
    loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
)
gan.fit(
    dataset, epochs=epochs, callbacks=[GANMonitor(num_img=3, latent_dim=latent_dim)]
)
