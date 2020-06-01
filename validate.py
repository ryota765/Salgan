import argparse

import numpy as np

from utils import metrics


def parse_args():
    parser = argparse.ArgumentParser(
        description='Parameters for saliency map generator models.')

    parser.add_argument('--gt_data_path', type=str,
                        help='path to .npy file of ground truth data', default='data/Yval.npy')
    parser.add_argument('--prediction_data_path', type=str,
                        help='path to .npy file of predicted results', default='prediction/pred.npy')

    return parser.parse_args()

def load_data(gt_path,prediction_path):
    gt = np.load(gt_path).astype(np.uint8)
    pred_result = np.load(prediction_path).astype(np.uint8)
    return gt, pred_result


if __name__ == '__main__':
    args = parse_args()

    gt_path = args.gt_data_path
    prediction_path = args.prediction_data_path

    gt_array, pred_array = load_data(gt_path, prediction_path)

    auc_b, auc_s, nss, cc = metrics.SaliencyMetrics().calculate_metrics(pred_array,gt_array)

    print('auc_b', auc_b)
    print('auc_s', auc_s)
    print('nss', nss)
    print('cc', cc)