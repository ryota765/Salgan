from keras.applications.vgg16 import VGG16
from keras.models import Sequential, Model, model_from_json
from keras.layers import Input, InputLayer, Activation, Dropout, Flatten, Dense, Conv2D, UpSampling2D, AveragePooling2D, MaxPooling2D
from keras import optimizers
import keras.backend as K
import tensorflow as tf
import keras.backend.tensorflow_backend as tfb
import keras
from keras.layers.merge import concatenate
from keras import regularizers

class ModelBuilder():
    '''Construct model for salgan and BCE
    '''

    @staticmethod
    def build_encoder(img_width,img_height,l2_norm):
        input_tensor = Input(shape=(img_width, img_height, 3))
        # vgg16 = VGG16(include_top=False, weights='model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', input_tensor=input_tensor)
        vgg16 = VGG16(include_top=False, weights='imagenet', input_tensor=input_tensor)

        model_encoder = Sequential()
        model_encoder.add(InputLayer(input_shape=(img_height,img_width, 3)))

        for i,layer in enumerate(vgg16.layers[:-1]):
            if i <= 10:
                layer.trainable = False
            else:
                layer.kernel_regularizer=regularizers.l2(l2_norm)
            model_encoder.add(layer)
        return model_encoder

    @staticmethod
    def build_decoder(img_width,img_height,l2_norm):
        model_decoder = Sequential()

        model_decoder.add(Conv2D(512,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(512,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(512,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(UpSampling2D((2,2)))

        model_decoder.add(Conv2D(512,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(512,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(512,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(UpSampling2D((2,2)))

        model_decoder.add(Conv2D(256,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(256,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(256,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(UpSampling2D((2,2)))

        model_decoder.add(Conv2D(128,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(128,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(UpSampling2D((2,2)))

        model_decoder.add(Conv2D(64,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(64,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_decoder.add(Conv2D(1,1,activation='sigmoid'))

        return model_decoder

    def generator(self,img_width,img_height,l2_norm=0,load_model_path=None):
        model_encoder = self.build_encoder(img_width,img_height,l2_norm)
        model_decoder = self.build_decoder(img_width,img_height,l2_norm)

        model_generator = Model(input=model_encoder.input, output=model_decoder(model_encoder.output))

        if load_model_path != None:
            print('Loading model weights from {}'.format(load_model_path))
            model_generator.load_weights(load_model_path)
        
        model_generator.summary()

        return model_generator

    @staticmethod
    def discriminator(img_width,img_height,l2_norm):
        model_discriminator = Sequential()

        model_discriminator.add(Conv2D(3,1,input_shape=(img_height,img_width,4),activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(Conv2D(32,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(MaxPooling2D((2,2)))

        model_discriminator.add(Conv2D(64,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(Conv2D(64,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(MaxPooling2D((2,2)))

        model_discriminator.add(Conv2D(64,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(Conv2D(64,3,activation='relu',padding='same',kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(MaxPooling2D((2,2)))

        model_discriminator.add(Flatten())
        model_discriminator.add(Dense(100,kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(Activation('tanh'))
        model_discriminator.add(Dense(2,kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(Activation('tanh'))
        model_discriminator.add(Dense(1,kernel_regularizer=regularizers.l2(l2_norm)))
        model_discriminator.add(Activation('sigmoid'))

        model_discriminator.summary()

        return model_discriminator

    @staticmethod
    def build_combine(generator, discriminator, img_width,img_height):
        inputs = Input(shape=(img_height,img_width,3))
        generated_images = generator(inputs)
        outputs = discriminator(concatenate([inputs,generated_images], axis=3))
        model = Model(inputs=inputs, outputs=[generated_images, outputs])
        return model

class LossFunction():
    '''Original BCE loss mentioned in paper
    1/4 downscaling using AveragePooling is conducted
    '''

    @staticmethod
    def binary_crossentropy_forth(y_true, y_pred):
        y_true_forth = AveragePooling2D(pool_size=(4, 4), padding='valid')(y_true)
        y_pred_forth = AveragePooling2D(pool_size=(4, 4), padding='valid')(y_pred)
        return K.mean(K.binary_crossentropy(y_true_forth, y_pred_forth), axis=-1)
