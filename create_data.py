import sys
import numpy as np
import os
from PIL import Image
from tqdm import tqdm
import random
#设置随机数种子
random.seed(2021)

def write_txt(file_name, imgs_path, labels_path=None, mode='train', val_pro=0.2):
    assert mode=="train" or mode=="test", "ERROR:mode must be train or test."
    if mode!="test":
        train_path = []
        for idx, f_path in enumerate(imgs_path):
            for i_path in sorted(os.listdir(f_path)):
                path1 = os.path.join(f_path, i_path) 
                path2 = os.path.join(labels_path[idx], i_path)
                train_path.append((path1, path2, str(idx)))
        
        if val_pro>=0 and val_pro<=1:
            #打乱数据
            random.shuffle(train_path)
            val_len = int(len(train_path)*val_pro)
            val_path = train_path[:val_len]
            train_path = train_path[val_len:]
            with open(file_name[0], 'w') as f:
                for path in train_path:
                    f.write(path[0]+" "+path[1]+" "+path[2]+"\n")
            with open(file_name[1], 'w') as f:
                for path in val_path:
                    f.write(path[0]+" "+path[1]+" "+path[2]+"\n")  
            return len(train_path), val_len
        else:
            with open(file_name[0], 'w') as f:
                for path in train_path:
                    f.write(path[0]+" "+path[1]+" "+path[2]+"\n") 
            return len(train_path), 0
    else:
        with open(file_name, 'w') as f:
            for path in imgs_path:
                img_path = os.path.join(test_path, path)
                f.write(img_path+"\n")


def create_txt(data_root, train_imgs_dir=None, train_labels_dir=None, test_dir=None, val_pro=0.2):
    if train_imgs_dir is not None:
        if os.path.exists("train.txt"):
            os.remove("train.txt")
        if os.path.exists("val.txt"):
            os.remove("val.txt")
        train_imgs_dir = os.path.join(data_root, train_imgs_dir)
        train_labels_dir = os.path.join(data_root, train_labels_dir)
        file_names = os.listdir(train_imgs_dir)
        file_names = sorted(file_names)
        train_imgs_path, train_labels_path =[], []
        for na in file_names:
            train_imgs_path.append(os.path.join(train_imgs_dir, na))
            train_labels_path.append(os.path.join(train_labels_dir, na))
        train_len, val_len = write_txt(["train.txt", "val.txt"], train_imgs_path, train_labels_path, mode='train', val_pro=val_pro)
        
        print("训练数据整理完毕！训练集长度：{}，验证集长度：{}， 类别数：{}".format(train_len, val_len, len(file_names)))

    if test_dir is not None:
        if os.path.exists("test.txt"):
            os.remove("test.txt")
        global test_path
        test_path = os.path.join(data_root, test_dir)
        test_imgs_path_list = sorted(os.listdir(test_path))
        write_txt("test.txt", test_imgs_path_list, mode="test")
        print("测试数据整理完毕！测试集长度：{}".format(len(test_imgs_path_list)))


data_root = "data"
train_imgs_dir = "train_image"
train_labels_dir = "train_50k_mask"
test_dir = "val_image"
create_txt(data_root, train_imgs_dir, train_labels_dir, test_dir, val_pro=0.2)


