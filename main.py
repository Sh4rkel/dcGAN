import tensorflow as tf
from tensorflow.keras import layers
import streamlit as st
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import os

def build_generator():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(100,)),
        layers.Dense(8 * 8 * 256, use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Reshape((8, 8, 256)),
        layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False),
        layers.BatchNormalization(),
        layers.LeakyReLU(),

        layers.Conv2DTranspose(3, (5, 5), strides=(1, 1), padding='same', use_bias=False, activation='tanh')
    ])
    return model

def build_discriminator():
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(32, 32, 3)),
        layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'),
        layers.LeakyReLU(),
        layers.Dropout(0.3),

        layers.Flatten(),
        layers.Dense(1)
    ])
    return model

def load_pixel_art_images(directories, image_size=(32, 32)):
    images = []
    for folder in directories:
        if not os.path.exists(folder):
            print(f"Directory {folder} does not exist. Skipping...")
            continue
        for filename in os.listdir(folder):
            img = Image.open(os.path.join(folder, filename)).convert('RGB')
            img = img.resize(image_size, Image.Resampling.LANCZOS)
            img = np.array(img) / 127.5 - 1
            images.append(img)
    return np.array(images)

directories = ['data/Cars Dataset/train/Audi',
               'data/Cars Dataset/train/Hyundai Creta',
               'data/Cars Dataset/train/Mahindra Scorpio',
               'data/Cars Dataset/train/Rolls Royce',
               'data/Cars Dataset/train/Swift',
               'data/Cars Dataset/train/Tata Safari',
               'data/Cars Dataset/train/Toyota Innova',]
pixel_art_images = load_pixel_art_images(directories)

buffer_size = len(pixel_art_images)
batch_size = 256
train_dataset = tf.data.Dataset.from_tensor_slices(pixel_art_images).shuffle(buffer_size).batch(batch_size)

generator = build_generator()
discriminator = build_discriminator()

dummy_noise = tf.random.normal([1, 100])
_ = generator(dummy_noise)

dummy_image = tf.random.normal([1, 32, 32, 3])
_ = discriminator(dummy_image)

generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)

@tf.function
def train_step(images):
    noise = tf.random.normal([batch_size, 100])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator(noise, training=True)

        real_output = discriminator(images, training=True)
        fake_output = discriminator(generated_images, training=True)

        gen_loss = generator_loss(fake_output)
        disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))

def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)

def discriminator_loss(real_output, fake_output):
    real_loss = cross_entropy(tf.ones_like(real_output), real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    return real_loss + fake_loss

def train(dataset, epochs):
    for epoch in range(epochs):
        for image_batch in dataset:
            train_step(image_batch)

st.title("Pixel Art Generator")

seed = tf.random.normal([16, 100])

if st.button('Generate Pixel Art'):
    generated_images = generator(seed, training=False)
    save_dir = 'save'
    os.makedirs(save_dir, exist_ok=True)

    fig = plt.figure(figsize=(4, 4))
    for i in range(generated_images.shape[0]):
        img_array = (generated_images[i] * 127.5 + 127.5).numpy().astype(np.uint8)
        img = Image.fromarray(img_array, 'RGB')
        img.save(os.path.join(save_dir, f'generated_image_{i}.png'))

        plt.subplot(4, 4, i + 1)
        plt.imshow(img_array)
        plt.axis('off')

    st.pyplot(fig)

    print(f"Generated images shape: {generated_images.shape}")
    print(f"Generated images range: {generated_images.numpy().min()} to {generated_images.numpy().max()}")
    print(f"Generated images (raw) range: {generated_images.min()} to {generated_images.max()}")

    if np.any(np.isnan(generated_images.numpy())) or np.any(np.isinf(generated_images.numpy())):
        print("Warning: NaN or Inf values found in generated images")

    for i in range(generated_images.shape[0]):
        plt.subplot(4, 4, i + 1)
        plt.imshow((generated_images[i] + 1) / 2)
        plt.axis('off')
    plt.show()

    intermediate_layer_model = tf.keras.Model(inputs=generator.input, outputs=generator.layers[-2].output)
    intermediate_output = intermediate_layer_model(seed)
    print(f"Intermediate output shape: {intermediate_output.shape}")
    print(f"Intermediate output range: {intermediate_output.numpy().min()} to {intermediate_output.numpy().max()}")