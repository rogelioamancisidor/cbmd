# -*- coding: utf-8 -*-
from losses import log_diag_mvn, kld_unit_mvn, binary_crossentropy, loss_q_logp, compute_mmd, log_normal_pdf
from Layers import EncoderGaussian, DecoderBernoulli, CLS, DecoderGaussian
import tensorflow as tf 
import numpy as np

class CBMD(tf.keras.Model):
    def __init__(self,
                dim2,
                ydim,
                lambda_val, 
                omega, 
                alpha,
                layers_size_enc  = [1024,1024,1024],
                layers_size_enc2 = [1024,1024,1024],
                layers_size_dec  = [1024,1024,1024],
                layers_size_prior = [1024,1024,1024],
                layers_size_cls= [50,50],
                dropout_enc = [0.2,0.2,0.2],
                dropout_enc2= [0.2,0.2,0.2],
                dropout_dec = [0.2,0.2,0.2],
                dropout_prior = [0.2,0.2,0.2],
                dropout_cls = [0.2,0.2],
                latent_dim = 30,
                n_samples = 1,
                name ='cbmd',
                decoder_type ='Bernoulli',
                **kwargs):
        super(CBMD, self).__init__(name=name, **kwargs)
        self.lambda_val =lambda_val 
        self.omega = omega 
        self.alpha = alpha
        self.ydim = ydim
        
        # q(z|x1,x2,y)
        self.encoder = EncoderGaussian(latent_dim, layers_size=layers_size_enc, dropout_rates=dropout_enc, n_samples=n_samples)
        
        # q(z|x2)
        self.encoder2 = EncoderGaussian(latent_dim, layers_size=layers_size_enc2, dropout_rates=dropout_enc2, n_samples=n_samples)
        
        # p(x2|x1,z)
        self.decoder_type = decoder_type
        if self.decoder_type == 'Bernoulli':
            self.decoder = DecoderBernoulli(dim2, layers_size=layers_size_dec, dropout_rates=dropout_dec)
        elif self.decoder_type == 'Gaussian':
            self.decoder = DecoderGaussian(dim2, layers_size=layers_size_dec, dropout_rates=dropout_dec)

        # p(z|x1)
        self.prior = EncoderGaussian(latent_dim, layers_size=layers_size_prior, dropout_rates=dropout_prior, n_samples=n_samples)

        # q(y|z)
        # same droprate for all networks
        self.cls = CLS(ydim, layers_size=layers_size_cls, dropout_rates=dropout_cls)

    def call(self, inputs, training=True):
        # loss function for classifier
        x1, x2, y = inputs
        # q(z|x1,x2,y)
        input_to_encoder = tf.concat([x1,x2,y],1)
        zpost_mean, zpost_log_var, zpost    = self.encoder(input_to_encoder, training=training)
        
        # p(z|x1)
        zprior_mean, zprior_log_var, zprior = self.prior(x1, training=training)

        # p(x2|z,x1)
        _recon_loss = 0 
        if self.decoder_type == 'Bernoulli':
            for i in range(len(zpost)):
                input_to_decoder = tf.concat([zpost[i],x1],1)
                x2_hat = self.decoder(input_to_decoder, training=training)
                _recon_loss += binary_crossentropy(x2,x2_hat)
        elif self.decoder_type == 'Gaussian': 
            for i in range(len(zpost)):
                input_to_decoder = tf.concat([zpost[i],x1],1)
                x2_mu, x2_log_var, x2_hat = self.decoder(input_to_decoder, training=training)
                _recon_loss += log_normal_pdf(x2, x2_mu, x2_log_var)
        recon_loss = tf.reduce_mean(_recon_loss)
        
        # q(z|x2)
        # XXX stop this forward pass from being backpropagated
        input_to_decoder = tf.concat([zprior_mean,x1],1)
        x2_mu_p,_,_ = self.decoder(input_to_decoder, training=training)
        x2_mu_p = tf.stop_gradient(x2_mu_p)
        
        zx2_mean, zx2_log_var, zx2 = self.encoder2(x2_mu_p, training=training)
        
        input_to_cls = tf.concat([zprior_mean, zx2_mean],1)
        pi_hat = self.cls(input_to_cls, training=training)
        
        _mmd  = 0
        _mmd2 = 0
        for i in range(len(zpost)):
            _mmd  += compute_mmd(zprior[i], zpost[i])
            _mmd2 += compute_mmd(zx2[i], zprior[i])
        mmd  = self.lambda_val * _mmd/len(zpost)
        mmd2 = self.lambda_val * _mmd2/len(zpost)
        
        kl_loss  = loss_q_logp(zpost_mean, tf.math.exp(zpost_log_var), zprior_mean, tf.math.exp(zprior_log_var))
        kl_loss2 = kld_unit_mvn(zx2_mean, zx2_log_var)
                
        cls_loss = tf.reduce_mean(self.alpha*tf.nn.softmax_cross_entropy_with_logits(y, pi_hat))
        
        cost1 = kl_loss + recon_loss + cls_loss + kl_loss2 + mmd2 
        cost2 = recon_loss + cls_loss + kl_loss2 + mmd + mmd2
    
        self.cbmd_loss =  self.omega*cost1 + (1-self.omega)*cost2  
        
        self.params = self.decoder.trainable_variables + self.encoder.trainable_variables\
                      + self.prior.trainable_variables + self.cls.trainable_variables\
                      + self.encoder2.trainable_variables  
        
        
        return {'kl':kl_loss,'recon_loss': recon_loss, 'cls_loss':cls_loss}

    def generate(self, x1, training=False):
        zprior_mean, _ , _ = self.prior(x1, training=training)

        # XXX at test x2 is not available so we use zprior for generating x2 view
        input_to_decoder = tf.concat([zprior_mean,x1],1)
        
        if self.decoder_type == 'Bernoulli':
            x2_mu = self.decoder(input_to_decoder, training=training)
        elif self.decoder_type == 'Gaussian':
            x2_mu,_,_ = self.decoder(input_to_decoder, training=training)
        
        return {'x2_hat': x2_mu}
    
    def classify(self, x1, training=False):
        zprior_mean, _, _ = self.prior(x1, training=training)
        input_to_decoder = tf.concat([zprior_mean,x1],1)
        
        if self.decoder_type == 'Bernoulli':
            x2_hat = self.decoder(input_to_decoder, training=training)
        elif self.decoder_type == 'Gaussian':
            x2_hat,_,_ = self.decoder(input_to_decoder, training=training)
        
        zx2_mean,_,_ = self.encoder2(x2_hat, training=training)
        
        input_to_cls = tf.concat([zprior_mean, zx2_mean],1)
        pi_hat = self.cls(input_to_cls, training=training)
        y_hat  = tf.one_hot(tf.math.argmax(pi_hat,1),self.ydim) 
        
        return {'pi_hat': pi_hat, 'y_hat':y_hat}
    
    @tf.function
    def train(self, x, optimizer):
        with tf.GradientTape() as tape:
            costs = self.call(x)
        gradients = tape.gradient(self.cbmd_loss, self.params)
        optimizer.apply_gradients(zip(gradients, self.params))

        return costs, self.cbmd_loss
