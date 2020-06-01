import numpy as np
import argparse

from utils import model

def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for saliency map generator models.')

    parser.add_argument('--target_data_path', type=str,
                        help='path to .npy file of input data', default='data/Xval.npy')
    parser.add_argument('--output_data_path', type=str,
                        help='path to output prediction in .npy file', default='prediction/pred.npy')
    parser.add_argument('--load_model_path', type=str,
                        help='Model path for using pre-trained model', default='model/weights_bce4_15.hdf5')
    parser.add_argument('--image_size', type=list,
                        help='size of input image [width,height]', default=[256,192])

    return parser.parse_args()

def predict(args):
    #-- parse parameters --#
    target_data_path = args.target_data_path
    output_data_path = args.output_data_path
    load_model_path = args.load_model_path
    img_width, img_height = args.image_size
    #-- parse parameters --#

    input_array = np.load(target_data_path).astype(np.float32)
    input_array /= 255

    model_builder = model.ModelBuilder()
    model_generator = model_builder.generator(img_width=img_width,img_height=img_height)
    model_generator.load_weights(load_model_path)

    result_array = model_generator.predict(input_array)
    result_array *= 255
    result_array = result_array.reshape((result_array.shape[0],result_array.shape[1],result_array.shape[2]))

    np.save(output_data_path,result_array.astype(np.uint8))

if __name__ == '__main__':
    args = parse_args()
    predict(args)