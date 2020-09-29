import tensorflow as tf
#from tqdm import tdqm
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1DTranspose, Conv1D, Reshape, Input, ReLU, UpSampling1D, LeakyReLU
import json
import time

generator = tf.keras.models.load_model('models/gen_at_epoch_200.h5',compile = False)
discriminator = tf.keras.models.load_model('models/disc_at_epoch_200.h5',compile = False)
generator.summary()

from scipy.io.wavfile import write as wavwrite

seed = tf.random.normal([1, 100])
#print(seed)
pred = generator(seed, training = False)
print(discriminator(pred))
pred = np.squeeze(pred.numpy(), axis =2)
wavwrite("preview.wav", 16000, pred.T)
