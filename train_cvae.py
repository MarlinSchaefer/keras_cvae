import os
main_dir = '.'
if os.path.isdir(main_dir):
    os.chdir(main_dir)
from CVAE import CVAE, load_cvae, safe_abs, SMALL_CONSTANT, GAUSS_RANGE
import tensorflow as tf
import tensorflow_probability as tfp
tfd = tfp.distributions
from tensorflow import keras
import tensorflow.keras.backend as K
from generator import SignalHandler, NoiseHandler, MultiHandler
from BnsLib.network.generators import FileGenerator
import numpy as np

#def von_mises_fisher_func(mean_direction, concentration):
    #return tfd.VonMisesFisher(mean_direction=tf.math.l2_normalize(mean_direction, axis=-1),
                              #concentration=tf.squeeze(safe_abs(concentration)),
                              #validate_args=True)

#dist_dict = {'valid_trunc': lambda loc, scale: tfd.TruncatedNormal(loc=loc, scale=safe_abs(scale), low=-10., high=11., validate_args=True),
             #'m1': lambda loc, scale: tfd.TruncatedNormal(loc=loc, scale=safe_abs(scale), low=34.9, high=80.1, validate_args=True),
             #'m2': lambda loc, scale: tfd.TruncatedNormal(loc=loc, scale=safe_abs(scale), low=34.9, high=80.1, validate_args=True),
             #'m2_cond': lambda loc, scale: lambda m1: tfd.TruncatedNormal(loc=loc, scale=safe_abs(scale), low=-10., high=10.+m1, validate_args=True),
             #'dist': lambda loc, scale: tfd.TruncatedNormal(loc=loc, scale=safe_abs(scale), low=999.9, high=3000.1, validate_args=True),
             #'time': lambda loc, scale: tfd.TruncatedNormal(loc=loc, scale=safe_abs(scale), low=0.14, high=0.36, validate_args=True),
             #'inc': lambda loc, scale: tfd.TruncatedNormal(loc=loc, scale=safe_abs(scale), low=-0.1, high=np.pi+0.1, validate_args=True),
             #'von_mises': lambda loc, concentration: tfd.VonMises(loc=loc, concentration=safe_abs(concentration), validate_args=True),
             #'von_mises_fisher': von_mises_fisher_func}

class CustomVonMisesFisher(tfd.VonMisesFisher):
    def log_prob(self, inp):
        ret = super().log_prob(inp)
        return tf.expand_dims(ret, axis=-1)

def von_mises_fisher_func(mean_direction, concentration, ramp=1.0):
    return CustomVonMisesFisher(mean_direction=tf.math.l2_normalize(mean_direction, axis=-1),
                                concentration=tf.squeeze(safe_abs(concentration)),
                                validate_args=True)

def trunc_gauss_func(loc, scale, ramp=1.0):
    sig = tf.sqrt(SMALL_CONSTANT + tf.exp(scale))
    low = -GAUSS_RANGE * (1.0 - ramp)
    high = GAUSS_RANGE * (1.0 - ramp) + 1.0
    return tfd.TruncatedNormal(loc=loc, scale=sig, low=low, high=high,
                               validate_args=True)

def m2_cond_func(loc, scale, ramp=1.0):
    sig = tf.sqrt(SMALL_CONSTANT + tf.exp(scale))
    low = -GAUSS_RANGE * (1.0 - ramp)
    high = GAUSS_RANGE * (1.0 - ramp)
    return lambda m1: tfd.TruncatedNormal(loc=loc, scale=sig, low=low,
                                          high=high + ramp * m1,
                                          validate_args=True)

dist_dict = {'valid_trunc': trunc_gauss_func,
             'm2_cond': lambda loc, scale, ramp: m2_cond_func(loc, scale, ramp),
             'von_mises': lambda loc, concentration: tfd.VonMises(loc=loc, concentration=safe_abs(concentration), validate_args=True),
             'von_mises_fisher': von_mises_fisher_func}

#output_dists = [('m1', ['loc', 'scale']),
                #('m2', ['loc', 'scale']),
                #('dist', ['loc', 'scale']),
                #('time', ['loc', 'scale']),
                #('inc', ['loc', 'scale']),
                #('von_mises_fisher', [('mean_direction', (3,)), ('concentration', (1,))])]
output_dists = [('valid_trunc', ['loc', 'scale']),
                ('m2_cond', ['loc', 'scale']),
                ('valid_trunc', ['loc', 'scale']),
                ('valid_trunc', ['loc', 'scale']),
                ('valid_trunc', ['loc', 'scale']),
                ('von_mises_fisher', [('mean_direction', (3,)), ('concentration', None)])]

def get_E1():
    E1inp = keras.layers.Input(shape=(256, 3))
    
    E1conv1 = keras.layers.Conv1D(33, 5)(E1inp)
    E1act1 = keras.layers.Activation('relu')(E1conv1)
    
    E1conv2 = keras.layers.Conv1D(33, 8)(E1act1)
    E1pool2 = keras.layers.MaxPooling1D(2, strides=2)(E1conv2)
    E1act2 = keras.layers.Activation('relu')(E1pool2)
    
    E1conv3 = keras.layers.Conv1D(33, 11)(E1act2)
    E1act3 = keras.layers.Activation('relu')(E1conv3)
    
    E1conv4 = keras.layers.Conv1D(33, 10)(E1act3)
    E1pool4 = keras.layers.MaxPooling1D(2, strides=2)(E1conv4)
    E1act4 = keras.layers.Activation('relu')(E1pool4)
    
    E1conv5 = keras.layers.Conv1D(33, 10)(E1act4)
    E1act5 = keras.layers.Activation('relu')(E1conv5)
    E1flat5 = keras.layers.Flatten()(E1act5)
    
    E1fc6 = keras.layers.Dense(2048)(E1flat5)
    E1drop6 = keras.layers.Dropout(0.2)(E1fc6)
    E1act6 = keras.layers.Activation('relu')(E1drop6)
    
    E1fc7 = keras.layers.Dense(2048)(E1act6)
    E1drop7 = keras.layers.Dropout(0.2)(E1fc7)
    E1act7 = keras.layers.Activation('relu')(E1drop7)
    
    E1fc8 = keras.layers.Dense(10 * 16)(E1act7)
    E1out1 = keras.layers.Reshape((16, 10))(E1fc8)
    
    E1fc9 = keras.layers.Dense(10 * 16)(E1act7)
    E1out2 = keras.layers.Reshape((16, 10))(E1fc9)
    
    E1out3 = keras.layers.Dense(16)(E1act7)
    
    return keras.models.Model(inputs=[E1inp], outputs=[E1out1, E1out2, E1out3])

def get_E2():
    E2inp1 = keras.layers.Input(shape=(8,))
    E2inp2 = keras.layers.Input(shape=(256, 3))
    
    E2conv1 = keras.layers.Conv1D(33, 5)(E2inp2)
    E2act1 = keras.layers.Activation('relu')(E2conv1)
    
    E2conv2 = keras.layers.Conv1D(33, 8)(E2act1)
    E2pool2 = keras.layers.MaxPooling1D(2, strides=2)(E2conv2)
    E2act2 = keras.layers.Activation('relu')(E2pool2)
    
    E2conv3 = keras.layers.Conv1D(33, 11)(E2act2)
    E2act3 = keras.layers.Activation('relu')(E2conv3)
    
    E2flat = keras.layers.Flatten()(E2act3)
    E2conc = keras.layers.concatenate([E2flat, E2inp1])
    
    E2fc4 = keras.layers.Dense(2048)(E2conc)
    E2drop4 = keras.layers.Dropout(0.2)(E2fc4)
    E2act4 = keras.layers.Activation('relu')(E2drop4)
    
    E2fc5 = keras.layers.Dense(2048)(E2act4)
    E2drop5 = keras.layers.Dropout(0.2)(E2fc5)
    E2act5 = keras.layers.Activation('relu')(E2drop5)
    
    E2out1 = keras.layers.Dense(10)(E2act5)
    E2out2 = keras.layers.Dense(10)(E2act5)
    
    return keras.models.Model(inputs=[E2inp1, E2inp2], outputs=[E2out1, E2out2])

def get_D():
    Dinp1 = keras.layers.Input(shape=(256, 3))
    Dinp2 = keras.layers.Input(shape=(10, ))
    
    Dconv1 = keras.layers.Conv1D(33, 5)(Dinp1)
    Dact1 = keras.layers.Activation('relu')(Dconv1)
    
    Dconv2 = keras.layers.Conv1D(33, 8)(Dact1)
    Dpool2 = keras.layers.MaxPooling1D(2, strides=2)(Dconv2)
    Dact2 = keras.layers.Activation('relu')(Dpool2)
    
    Dconv3 = keras.layers.Conv1D(33, 11)(Dact2)
    Dact3 = keras.layers.Activation('relu')(Dconv3)
    
    Dflat = keras.layers.Flatten()(Dact3)
    Dconc = keras.layers.concatenate([Dflat, Dinp2])
    
    Dfc4 = keras.layers.Dense(2048)(Dconc)
    Ddrop4 = keras.layers.Dropout(0.2)(Dfc4)
    Dact4 = keras.layers.Activation('relu')(Ddrop4)
    
    Dfc5 = keras.layers.Dense(2048)(Dact4)
    Ddrop5 = keras.layers.Dropout(0.2)(Dfc5)
    Dact5 = keras.layers.Activation('relu')(Ddrop5)
    
    Dout_loc = keras.layers.Dense(7, activation='sigmoid')(Dact5)
    Dout_var = keras.layers.Dense(7, activation='relu')(Dact5)
    
    Dstack = tf.stack([Dout_loc, -Dout_var], axis=-1)
    Dout = keras.layers.Reshape((14,))(Dstack)
    
    return keras.models.Model(inputs=[Dinp1, Dinp2], outputs=[Dout])

def get_cvae():
    return CVAE(get_E1(), get_E2(), get_D(), output_dists,
                dist_dict=dist_dict)

def get_handler():
    data_dir = 'training_data_corr'
    files = os.listdir(os.path.join(main_dir, data_dir))
    
    multi_handeler = MultiHandler()
    ref_idx = 0
    
    for fn in files:
        path = os.path.join(main_dir, data_dir, fn)
        sig_handler = SignalHandler(path, ref_index=ref_idx)
        multi_handeler.add_file_handeler(sig_handler, group='signal')
        ref_idx += len(sig_handler)
    
    return multi_handeler

class ModelSaver(keras.callbacks.ModelCheckpoint):
    def set_model(self, model):
        self.model = model

class RampSetter(tf.keras.callbacks.Callback):
    def __init__(self, start, stop, **kwargs):
        super().__init__(**kwargs)
        self.start = start
        self.stop = stop
        assert self.stop > self.start
        self.log_diff = np.log10(self.stop) - np.log10(self.start)
        self.batches = 0
        self.ramp = K.epsilon()
    
    def on_train_batch_begin(self, batch, logs=None):
        self.model.set_ramp(max(K.epsilon(), min(1.0, abs(self.ramp))))
    
    def on_train_batch_end(self, batch, logs=None):
        if self.batches > self.stop:
            return
        self.batches += 1
        if self.batches >= self.start:
            self.ramp = (np.log10(float(self.batches)) - np.log10(self.start)) / self.log_diff
            

def main():
    save_path = 'custom_cvae_compare'
    num_repeats = 2
    cvae = get_cvae()
    handler = get_handler()
    with handler as h:
        gen = FileGenerator(h, np.repeat(np.arange(len(h)), num_repeats),
                            shuffle=True, batch_size=64)
        #gen = FileGenerator(h, np.repeat(np.arange(64+16), num_repeats),
                            #shuffle=True, batch_size=64)
        checkpoint_path = os.path.join(main_dir, save_path, 'model_{epoch:d}')
        check = ModelSaver(checkpoint_path,
                            verbose=1,
                            save_freq='epoch',
                            save_best_only=False,
                            save_weights_only=False)
        ramp = RampSetter(start=10000, stop=100000)
        csv_path = os.path.join(main_dir, save_path, 'loss_history.csv')
        logger = keras.callbacks.CSVLogger(csv_path)
        opti = keras.optimizers.Adam(learning_rate=1e-5)
        cvae.compile(optimizer=opti, loss=None)
        cvae.fit(gen, epochs=100, callbacks=[check, logger, ramp])
        #cvae.fit(gen, epochs=10, callbacks=[check])
        
    return

if __name__ == "__main__":
    main()
