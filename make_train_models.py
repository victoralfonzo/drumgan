import tensorflow as tf
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Conv1DTranspose, Conv1D, Reshape, Input, ReLU, UpSampling1D, LeakyReLU
import json
import time
import sys
from scipy.io.wavfile import write as wavwrite

tf.compat.v1.enable_eager_execution
cross_entropy_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)

EPOCHS = 20000

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

def generator_loss(fake_output, which):
    if (which !='simple'):
        return -tf.reduce_mean(fake_output)
    return cross_entropy_fn(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_sample, fake_sample, which):
    if (which!='simple'):
        #print(real_sample, fake_sample)
        real_loss = tf.reduce_mean(real_sample)
        fake_loss = tf.reduce_mean(fake_sample)
        #print('real_loss')
        #print(real_loss)
        #print('fake_loss')
        #print(fake_loss)
        loss = fake_loss - real_loss
        #print('loss')
        #print(loss.shape)
        return loss
    real_loss = cross_entropy_fn(tf.ones_like(real_sample), real_sample)
    fake_loss = cross_entropy_fn(tf.zeros_like(fake_sample), fake_sample)
    return real_loss + fake_loss

def gradient_penalty(batch_size, real_samples, fake_samples):
    alpha = tf.random.uniform(shape = [batch_size, 1,1], minval = 0., maxval = 1.)
    #print(real_samples.shape)
    #print(fake_samples.shape)
    
    diff = fake_samples - real_samples
    interp = real_samples + (alpha * diff)
    with tf.GradientTape() as gp_tape:
        gp_tape.watch(interp)
        pred = discriminator(interp, training = True)
    gradients = gp_tape.gradient(pred, [interp])[0]
    norm = tf.sqrt(tf.reduce_sum(tf.square(gradients), axis  = [1,2]))
    return tf.reduce_mean((norm-1.0) ** 2)

@tf.function
def train_step(real_samples, which,dpg = 5):
    batch_size = 64
    if(which != 'simple'):
        for _ in range(dpg):
            random_latent_vectors = tf.random.normal(shape = (batch_size, 100))
            with tf.GradientTape() as tape:
                fake_samples = generator(random_latent_vectors, training = True)
                fake_logits = discriminator(fake_samples, training = True)
                real_logits = discriminator(real_samples, training = True)
                d_loss = discriminator_loss(real_logits, fake_logits, 'gp')
                
                gp = gradient_penalty(batch_size, real_samples, fake_samples)
                d_loss = d_loss + (gp * 10)
                
            d_gradients = tape.gradient(d_loss, discriminator.trainable_variables)
            discriminator_optimizer.apply_gradients(zip(d_gradients, discriminator.trainable_variables))
        

        random_latent_vectors = tf.random.normal(shape = (batch_size, 100))
        with tf.GradientTape() as tape:
            gen_samples = generator(random_latent_vectors, training = True)
            gen_logits = discriminator(gen_samples, training = True)
            g_loss = generator_loss(gen_logits, 'gp')

        g_gradients = tape.gradient(g_loss, generator.trainable_variables)
        generator_optimizer.apply_gradients(zip(g_gradients, generator.trainable_variables))
        print("Discriminator Loss:")
        tf.print(d_loss, output_stream=sys.stderr)
        print(" Generator Loss:")
        tf.print(g_loss, output_stream=sys.stderr)
        return {"d_loss":d_loss, "g_loss": g_loss}
    else:
        noise = tf.random.normal(shape = (batch_size, 100))
        with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
            generated_samples = generator(noise, training = True)
            real_out = discriminator(real_samples, training = True)
            fake_out = discriminator(generated_samples, training = True)

            gen_loss = generator_loss(fake_out, 'b')
            disc_loss = discriminator_loss(real_out, fake_out,'b')

        gen_gradients = gen_tape.gradient(gen_loss, generator.trainable_variables)
        disc_gradients = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

        generator_optimizer.apply_gradients(zip(gen_gradients, generator.trainable_variables))
        discriminator_optimizer.apply_gradients(zip(disc_gradients, discriminator.trainable_variables))
        return {"d_loss":disc_loss, "g_loss": gen_loss}



def train(dataset, epochs):
    for epoch in range(epochs):
       
        print("epoch #", epoch)
        if epoch % 200 == 0:
            filename = "models/{}_at_epoch_{}.h5"
            disc_filename = filename.format("disc",epoch)
            gen_filename = filename.format("gen",epoch)
            discriminator.save(disc_filename)
            generator.save(gen_filename)
        for batch in dataset:
            res = train_step(batch, 'banana') 
        print("Discriminator Loss:")
        tf.print(res['d_loss'], output_stream=sys.stderr)
        print(" Generator Loss:")
        tf.print(res['g_loss'], output_stream=sys.stderr)
            
print("Loading data")
X = load_data("data.json")
print("Data loaded successfully")
squeezed = np.squeeze(X,axis = 2)
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
