import time
from operator import add
import torch
import torch.nn.functional as F
import sys
sys.path.append('./models')
import numpy as np
import os, argparse
import cv2
from models.MyNet import MyNet
from data_loader import test_dataset
from utils import calculate_metrics
from tqdm import tqdm
import torch
import gc

def get_score(metrics_score, test_len):
    f1 = metrics_score[0] / test_len
    iou = metrics_score[1] / test_len
    sm = metrics_score[2] / test_len
    em = metrics_score[3] / test_len
    wfm = metrics_score[4] / test_len

    return f1, iou, sm, em, wfm

def print_score(score, net_name, Dataset):
    first_row = "{:^20s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format(net_name, "Dice", "Iou",
                                                                                        "Smeasure",
                                                                                         "Emeasure", "wFmeasure", "M_FPS")
    second_row = "{:^20s}{:^15s}{:^15s}{:^15s}{:^15s}{:^15s}".format("----------", "----------","----------",
                                                                                   "----------", "----------",
                                                                                   "----------")
    with open("./log/score.txt", 'a') as f:
        f.write('\n' + first_row + '\n')
        f.write(second_row + '\n')
        for i in range(len(Dataset)):
            string = ''.join("{:^15.4f}".format(j) for j in score[i])
            f.write("{:^20s}".format(Dataset[i]) + string + '\n')
            print("{:^20s}".format(Dataset[i]) + string)

parser = argparse.ArgumentParser()
parser.add_argument('--testsize', type=int, default=352, help='testing size')
parser.add_argument('--test_path',type=str,default='./Test_data/',help='test dataset path')
opt = parser.parse_args()

dataset_path = opt.test_path
model = MyNet()
model.load_state_dict(torch.load('./cpt/S2Net.pth'))
model.cuda()
model.eval()
scores = []

test_datasets = ['Hospital_A', 'Hospital_B', 'Hospital_C','Hospital_D']
for dataset in test_datasets:
    save_path = './results/' + dataset + '/'
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    image_root = dataset_path + dataset + '/images/'
    gt_root = dataset_path + dataset + '/masks/'
    test_loader = test_dataset(image_root, gt_root, testsize=352)
    total_time = 0
    count = 0
    metrics_score_1 = [0.0, 0.0, 0.0, 0.0, 0.0]
    for i in tqdm(range(test_loader.size)):
        image, gt, name = test_loader.load_data()
        gt = np.asarray(gt, np.float32)
        gt /= (gt.max() + 1e-8)

        image = image.cuda()
        start_time = time.perf_counter()
        res, _, _, _ = model(image)
        end_time = time.perf_counter()
        count += 1
        total_time += end_time - start_time

        res = F.upsample(res, size=gt.shape, mode='bilinear', align_corners=False)
        res = res.sigmoid().data.cpu().numpy().squeeze()
        res = (res - res.min()) / (res.max() - res.min() + 1e-8)

        """ Evaluation metrics """
        score_1 = calculate_metrics(gt, res)
        metrics_score_1 = list(map(add, metrics_score_1, score_1))

        print('save img to: ', save_path + name)
        cv2.imwrite(os.path.join(save_path, name), res * 255)
    score = list(get_score(metrics_score_1, test_loader.size))
    fps = count / total_time
    print('FPS:', fps)
    print('Test Done!')
    scores.append(score)
print_score(scores, 'test', test_datasets)
