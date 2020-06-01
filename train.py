import os
import argparse

import numpy as np
import keras
from keras import optimizers

from utils import model


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for saliency map generator models.')

    parser.add_argument('--model_name', type=str,
                        help='name of model to use',
                        choices=[
                            "bce",
                            "salgan",])
    parser.add_argument('--data_path', type=str,
                        help='path to .npy file of input data', default='original_data')
    parser.add_argument('--num_epoch', type=int,
                        help='number of epochs', default=121)
    parser.add_argument('--batch_size', type=int,
                        help='size of batch', default=32)
    parser.add_argument('--l2_norm', type=float,
                        help='parameter for l2 norm', default=0.0001)
    parser.add_argument('--loss_alpha', type=float,
                        help='weight alpha for loss calculation', default=0.005)
    parser.add_argument('--image_size', type=list,
                        help='size of input image [width,height]', default=[256,192])
    parser.add_argument('--learning_rate', type=float,
                        help='learning rate for model training', default=3e-4)
    parser.add_argument('--model_save_ratio', type=int,
                        help='ratio for model saving', default=4)

    return parser.parse_args()


def load_data(model_name):
    # load data
    X_train = np.load('data_original/Xtrain.npy').astype(np.float32)
    Y_train = np.load('data_original/Ytrain.npy').astype(np.float32)

    X_train /= 255
    Y_train = Y_train.reshape(Y_train.shape[0],Y_train.shape[1],Y_train.shape[2],1)/255

    if model_name == 'salgan':
        return X_train, Y_train

    elif model_name == 'bce':

        X_val = np.load('data_original/Xval.npy')
        Y_val = np.load('data_original/Yval.npy')

        X_val /= 255
        Y_val = Y_val.reshape(Y_val.shape[0],Y_val.shape[1],Y_train.val[2],1)/255

        return X_train, Y_train, X_val, Y_val


def save_all_weights(epoch_number, g, d):
    g.save_weights('model/generator_{}.h5'.format(epoch_number))
    d.save_weights('model/discriminator_{}.h5'.format(epoch_number))

def train_bce(args):
    # parse parameters
    l2_norm = args.l2_norm
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    learning_rate = args.learning_rate
    img_width, img_height = args.image_size
    model_save_ratio = args.model_save_ratio

    X_train, Y_train, X_val, Y_val = load_data(args.model_name)

    model_builder = model.ModelBuilder()
    model_generator = model_builder.generator(img_width=img_width,img_height=img_height,l2_norm=l2_norm)

    model_generator.compile(loss=model.LossFunction().binary_crossentropy_forth, optimizer=optimizers.Adagrad(lr=learning_rate), metrics=['accuracy'])

    model_save_path = 'model/generator_bce_{epoch:02d}.h5'
    cp_cb = keras.callbacks.ModelCheckpoint(filepath=fpath, monitor='val_loss', verbose=1, period=model_save_ratio)

    model_generator.fit(x=X_train,y=Y_train,batch_size=batch_size,epochs=num_epoch,verbose=1,validation_data=(X_val, Y_val), callbacks=[cp_cb])


def train_salgan(args):
    # parse parameters
    l2_norm = args.l2_norm
    batch_size = args.batch_size
    num_epoch = args.num_epoch
    learning_rate = args.learning_rate
    img_width, img_height = args.image_size
    loss_alpha = args.loss_alpha
    model_save_ratio = args.model_save_ratio

    X_train, Y_train = load_data(args.model_name)
    
    model_builder = model.ModelBuilder()

    model_generator = model_builder.generator(img_width=img_width,img_height=img_height,l2_norm=l2_norm)
    model_discriminator = model_builder.discriminator(img_width=img_width,img_height=img_height,l2_norm=l2_norm)

    output_true_batch, output_false_batch = np.ones((batch_size, 1)), np.zeros((batch_size, 1))

    model_combine = model_builder.build_combine(model_generator,model_discriminator,img_width=img_width,img_height=img_height)

    model_discriminator.trainable = True
    model_discriminator.compile(optimizer=optimizers.Adagrad(lr=learning_rate), loss="binary_crossentropy")
    model_discriminator.trainable = False
    loss = [model.LossFunction().binary_crossentropy_forth, "binary_crossentropy"]
    loss_weights = [loss_alpha, 1]
    model_combine.compile(optimizer=optimizers.Adagrad(lr=learning_rate), loss=loss, loss_weights=loss_weights)
    model_discriminator.trainable = True

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

        if epoch % model_save_ratio == 0:
            save_all_weights(epoch, model_generator, model_discriminator)

        print("discriminator_loss", np.mean(d_losses), "combine_loss", np.mean(c_losses), "generator_loss", np.mean(g_losses))


if __name__ == '__main__':
    args = parse_args()
    model_name = args.model_name

    if model_name == 'salgan':
        train_salgan(args)
    else:
        train_bce(args)

# TODO
# train_BCEとtrain_salganに分けて引数で処理確認
# predictionのファイル作成
# validationのファイル作成