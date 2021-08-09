#encoding=utf-8
import os
from PIL import Image
import numpy as np
import cv2
import shutil


def checkCreatePath(path):
    if os.path.exists(path):
        print("删除已经存在的路径: {}".format(path))
        shutil.rmtree(path)
    os.makedirs(path)


def vote_hard(result_path, save_path):
    result_list = os.listdir(result_path)
    print(len(result_list))
    for img_name in os.listdir(os.path.join(result_path, result_list[0]), 'results'):
        label_path = os.path.join(result_path, result_list[0], 'results', img_name)
        label = np.asarray(Image.open(label_path))
        mask = np.zeros((label.shape[0], label.shape[1]), np.float32)
        for result_dir in result_list:
            label_path = os.path.join(result_path, result_dir, 'results', img_name)
            label = np.asarray(Image.open(label_path))
            mask += label
        mask_new = mask >= 255 * 3
        mask_new = mask_new.astype("float32") * 255
        mask = mask_new.astype("uint8")
        cv2.imwrite(os.path.join(save_path, img_name), mask)


if __name__ == '__main__':
    result_path = 'result_B/'
    save_path = 'results-9-3'
    assert result_path != save_path
    checkCreatePath(save_path)
    vote_hard(result_path, save_path, mode='train')



