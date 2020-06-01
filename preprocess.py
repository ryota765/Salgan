import argparse
import glob
import os

from PIL import Image
import numpy as np



def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameter for preprocessing.')

    parser.add_argument('--data_path', type=str,
                        help='path to .npy file of input data', default='data')
    parser.add_argument('--image_size', type=list,
                        help='size of input image [width,height]', default=[256,192])

    return parser.parse_args()


def preprocess(image_dir,save_dir,save_flag,img_width,img_height):

    X_list = []
    Y_list = []

    img_list = glob.glob(image_dir+'/*.jpg')

    for x_path in img_list:
        y_path = x_path.replace('images','maps').replace('jpg','png')

        # Confirm that same named file is also in maps directory
        try:
            x_img = np.array(Image.open(x_path).resize((img_width,img_height)))
            y_img = np.array(Image.open(y_path).resize((img_width,img_height)))
            if y_img.shape[:2]+(3,) == x_img.shape:
                X_list.append(x_img)
                Y_list.append(y_img)
        except:
            pass

    np.save(os.path.join(save_dir,'X{}.npy'.format(save_flag)), X_list)
    np.save(os.path.join(save_dir,'Y{}.npy'.format(save_flag)), Y_list)


def preprocess_test(image_dir,save_dir,save_flag,img_width,img_height):

    X_list = []

    img_list = glob.glob(image_dir+'/*.jpg')

    for x_path in img_list:
        x_img = np.array(Image.open(x_path).resize((img_width,img_height)))
        X_list.append(x_img)

    np.save(os.path.join(save_dir,'X{}.npy'.format(save_flag)), X_list)


if __name__ == '__main__':
    args = parse_args()
    data_path = args.data_path
    img_width, img_height = args.image_size

    print('Preprocessing train images and maps...')
    train_dir = os.path.join(data_path,'images/train')
    preprocess(train_dir,data_path,'train',img_width,img_height)

    print('Preprocessing validation images and maps...')
    val_dir = os.path.join(data_path,'images/val')
    preprocess(val_dir,data_path,'val',img_width,img_height)

    print('Preprocessing test images...')
    test_dir = os.path.join(data_path,'images/test')
    preprocess_test(test_dir,data_path,'test',img_width,img_height)
    
