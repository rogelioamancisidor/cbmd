# -*- coding: utf-8 -*-
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
import tensorflow as tf 
import numpy as np

class HiddenBlock(layers.Layer):
    def __init__(self, layers_size, dropout_rates, activation = 'softplus'):
        super(HiddenBlock, self).__init__()
        self.activation = activation

        nlayers = len(layers_size)
        dropout_rates = np.tile(dropout_rates, nlayers)
        _hiddenLayer = [] 
        _hiddenLayer.append(layers.Dense(layers_size[0],activation=self.activation))
        if dropout_rates[0] > 0:
            _hiddenLayer.append(layers.Dropout(dropout_rates[0]))

        for i in range(nlayers-1):
            _hiddenLayer.append(layers.Dense(layers_size[i+1],activation=self.activation))
            if dropout_rates[i+1] > 0:
                _hiddenLayer.append(layers.Dropout(dropout_rates[i+1]))
        self.hidden_layers = Sequential(_hiddenLayer)

    def call(self, inputs, training=True):
        # make the forward pass
        x = self.hidden_layers(inputs, training=training)
        return x

class Sampling(layers.Layer):
    """ Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""
    def call(self, inputs, n_samples = 1):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        samples = []

        for i in range(n_samples):
            epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
            sample = z_mean + tf.exp(0.5 * z_log_var) * epsilon
            samples.append(sample)
        
        return samples


class EncoderGaussian(layers.Layer):
    def __init__(self,
                latent_dim,
                layers_size=[64],
                dropout_rates=0,
                n_samples = 1,
                name='encoder',
                activation = 'softplus',
                **kwargs):
        super(EncoderGaussian, self).__init__(name=name, **kwargs)
        self.hidden_layers = HiddenBlock(layers_size, dropout_rates, activation=activation)
        self.mu = layers.Dense(latent_dim)
        self.log_var = layers.Dense(latent_dim)
        self.sampling = Sampling()
        self.n_samples = n_samples

    def call(self, inputs, training=True):
        x = self.hidden_layers(inputs,training=training)
        z_mean = self.mu(x)
        z_log_var = self.log_var(x)
        z = self.sampling((z_mean, z_log_var), n_samples = self.n_samples)

        return z_mean, z_log_var, z

# Use inheritance to define DecoderGaussian  
class DecoderGaussian(EncoderGaussian):
    def __init__(self,
                latent_dim,
                layers_size=[64],
                dropout_rates=0,
                n_samples=1,
                name='decoder',
                activation = 'softplus',
                **kwargs):
        EncoderGaussian.__init__(self,latent_dim,name=name, **kwargs)

class DecoderBernoulli(layers.Layer):
    def __init__(self,
                original_dim,
                layers_size=[64],
                dropout_rates=0,
                name='decoder',
                activation = 'softplus',
                **kwargs):
        super(DecoderBernoulli, self).__init__(name=name, **kwargs)
        self.hidden_layers = HiddenBlock(layers_size,dropout_rates, activation=activation)
        self.mu = layers.Dense(original_dim, activation='sigmoid')

    def call(self, inputs, training=True):
        x = self.hidden_layers(inputs, training=training)
        x_recon = self.mu(x)
        return x_recon

class CLS(layers.Layer):
    def __init__(self,
                y_dim,
                layers_size=[64],
                dropout_rates=0,
                name='cls',
                activation = 'softplus',
                **kwargs):
        super(CLS, self).__init__(name=name, **kwargs)
        self.hidden_layers = HiddenBlock(layers_size,dropout_rates,activation=activation)
        self.classifier = layers.Dense(y_dim, activation='softmax')

    def call(self, inputs, training=True):
        x = self.hidden_layers(inputs, training=training)
        pi_hat = self.classifier(x)
        return pi_hat
