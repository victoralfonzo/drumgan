import tensorflow as tf
#from tqdm import tdqm
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1DTranspose, Conv1D, Reshape, Input, ReLU, UpSampling1D, LeakyReLU
import json
import time
from scipy.io.wavfile import write as wavwrite

EPOCHS = 300

def makeGen():
    model = tf.keras.Sequential()
    model.add(Input(shape = (100,)))
    model.add(Dense(256, use_bias= True))
    model.add(Reshape((16,16)))
    model.add(ReLU())
    print("2",model.output_shape)

    model.add(Conv1DTranspose(64, kernel_size = 25,strides = 4, use_bias = True,padding ='same'))
    model.add(ReLU())
    print("3",model.output_shape)
    
    model.add(Conv1DTranspose(4, kernel_size = 25,strides = 4, use_bias = True,padding ='same'))
    model.add(ReLU())
    print("4",model.output_shape)

    model.add(Conv1DTranspose(2, kernel_size = 25,strides = 4, use_bias = True,padding ='same'))
    model.add(ReLU())
    print("5",model.output_shape)

    model.add(Conv1DTranspose(1, kernel_size = 25,strides = 4, use_bias = True,padding ='same'))
    model.add(ReLU())
    print("6",model.output_shape)

    model.add(Conv1DTranspose(1, kernel_size = 25,strides = 4, use_bias = True,padding ='same'))
    model.add(ReLU())
    print("7",model.output_shape)

    model.add(Activation('tanh'))
    return model

def makeDisc():
    model = tf.keras.Sequential()
    model.add(Input(shape = (16384,1)))
    print("0",model.output_shape)

    model.add(Conv1D(1,25,strides = 4, use_bias = True, padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    print("0",model.output_shape)
    model.add(Conv1D(2,25,strides = 4, use_bias = True, padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    print("0",model.output_shape)
    model.add(Conv1D(4,25,strides = 4, use_bias = True, padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    print("0",model.output_shape)
    model.add(Conv1D(8,25,strides = 4, use_bias = True, padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    print("0",model.output_shape)
    model.add(Conv1D(16,25,strides = 4, use_bias = True, padding = 'same'))
    model.add(LeakyReLU(alpha = 0.2))
    print("0",model.output_shape)
    model.add(Reshape((256,)))
    print("0",model.output_shape)
    model.add(Dense(1))
    print("0",model.output_shape)
    return model

def load_data(dataset_path):
    with open(dataset_path, "r") as fp:
        data = json.load(fp)
    X = np.array(data["snares"])
    return X.astype(dtype = np.float32)

def generator_loss(fake_output):
    return -tf.reduce_mean(fake_output)

def discriminator_loss(real_sample, fake_sample):
    real_loss = tf.reduce_mean(real_sample)
    fake_loss = tf.reduce_mean(fake_sample)
    return fake_loss - real_loss

def gradient_penalty(batch_size, real_samples, fake_samples):
    alpha = tf.random.uniform(shape = [batch_size, 1,1], minval = 0., maxval = 1.)
    print(real_samples.shape)
    print(fake_samples.shape)
    
    diff = fake_samples - real_samples
    interp = real_samples + (alpha * diff)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interp)
        pred = discriminator(interp, training = True)
    gradients = gp_tape.gradient(pred, [interp])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis  = [1,2]))
    return tf.reduce_mean((norm-1.0) ** 2)

@tf.function
def train_step(real_samples, dpg = 5):
    batch_size = 64
    for i in range(dpg):
        random_latent_vectors = tf.random.normal(shape = (batch_size, 100))
        with tf.GradientTape() as tape:
            fake_samples = generator(random_latent_vectors, training = True)
            #fake_samples = tf.dtypes.cast(fake_samples, tf.dtypes.float64)
            print("real shape",real_samples.shape)
            print("fake shape",fake_samples.shape)
            fake_logits = discriminator(fake_samples, training = True)
            real_logits = discriminator(real_samples, training = True)
            d_loss = discriminator_loss(real_logits, fake_logits)
            
            gp = gradient_penalty(batch_size, real_samples, fake_samples)
            d_loss = d_loss + gp * 10
        d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
        discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
    

    random_latent_vectors = tf.random.normal(shape = (batch_size, 100))
    with tf.GradientTape() as tape:
        gen_samples = generator(random_latent_vectors, training = True)
        gen_logits = discriminator(gen_samples, training = True)
        g_loss = generator_loss(gen_logits)

    g_gradients = tape.gradient(g_loss, generator.trainable_variables)
    generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))

    return {"d_loss":d_loss, "g_loss": g_loss}

def train(dataset, epochs):
    for epoch in range(epochs):
        print("epoch #", epoch)
        start = time.time()
        for batch in dataset:
            train_step(batch)
    
X = load_data("data.json")
print(X.dtype)
generator = makeGen()
discriminator = makeDisc()

generator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)
discriminator_optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4, beta_1=0.5, beta_2=0.9)

train_dataset = tf.data.Dataset.from_tensor_slices(squeezed).batch(64,drop_remainder=True)
train(train_dataset, EPOCHS)

seed = tf.random.normal([1, 100])
pred = generator(seed, training = False)
pred = np.squeeze(pred.numpy(), axis =2)
wavwrite("sample.wav", 16000, pred.T)