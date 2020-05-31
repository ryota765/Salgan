import os

import matplotlib.pyplot as plt
import numpy as np
from keras import optimizers

from utils import model

def load_data():
    # load data
    X_train = np.load('data_original/Xtrain.npy').astype(np.float32)
    Y_train = np.load('data_original/Ytrain.npy').astype(np.float32)

    X_train /= 255
    Y_train = Y_train.reshape(Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1)/255

    print(X_train.shape)
    print(Y_train.shape)

def save_all_weights(d, g, epoch_number):
    g.save_weights('model/generator_{}.h5'.format(epoch_number))
    d.save_weights('model/discriminator_{}.h5'.format(epoch_number))

def train():
    l2_norm = 0.0001
    batch_size = 32
    nb_epoch = 201

    model_builder = model.ModelBuilder()

    model_generator = model_builder.generator(img_width=256,img_height=192,l2_norm=l2_norm)
    model_discriminator = model_builder.discriminator(img_width=256,img_height=192,l2_norm=l2_norm)

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.zeros((batch_size, 1))

    model_combine = model_builder.build_combine(model_generator,model_discriminator,img_width=256,img_height=192)

    model_discriminator.trainable = True
    model_discriminator.compile(optimizer=optimizers.Adagrad(lr=3e-4), loss="binary_crossentropy")
    model_discriminator.trainable = False
    loss = [model.LossFunction().binary_crossentropy_forth, "binary_crossentropy"]
    loss_weights = [0.005, 1]
    model_combine.compile(optimizer=optimizers.Adagrad(lr=3e-4), loss=loss, loss_weights=loss_weights)
    model_discriminator.trainable = True

    c_loss_list = []
    d_loss_list = []

    for epoch in range(1,nb_epoch):
        print('epoch: {}/{}'.format(epoch, nb_epoch))
        print('batches: {}'.format(int(X_train.shape[0] / batch_size)))
        
        permutated_indexes = np.random.permutation(X_train.shape[0])

        d_losses = []
        g_losses = []
        c_losses = []

        for index in range(int(X_train.shape[0] / batch_size)):
            batch_indexes = permutated_indexes[index*batch_size:(index+1)*batch_size]
            image_batch = X_train[batch_indexes]
            salmap_batch = Y_train[batch_indexes]

            generated_salmap = model_generator.predict(x=image_batch, batch_size=batch_size)

            d_loss_real = model_discriminator.train_on_batch(np.concatenate([image_batch,salmap_batch], 3), output_true_batch)
            d_loss_fake = model_discriminator.train_on_batch(np.concatenate([image_batch,generated_salmap], 3), output_false_batch)
            d_loss = 0.5 * np.add(d_loss_fake, d_loss_real)
            d_losses.append(d_loss)

            model_discriminator.trainable = False

            c_loss = model_combine.train_on_batch(image_batch, [salmap_batch,output_true_batch])
            c_losses.append(c_loss[0])
            g_losses.append(c_loss[1])
            
            model_discriminator.trainable = True

        if epoch % 4 == 0:
            save_all_weights(model_discriminator, model_generator, epoch)

        print("discriminator_loss", np.mean(d_losses), "combine_loss", np.mean(c_losses), "generator_loss", np.mean(g_losses))

        c_loss_list.append(c_losses)
        d_loss_list.append(d_losses)

        sample_img = model_generator.predict(np.array([X_train[0]]))
        plt.imsave('result_image/{}.jpg'.format(epoch),sample_img.reshape(192,256))

train()