import torch.utils.data as data
from PIL import Image
import pandas as pd
import os
import image_utils as util
import random
import glob
import numpy as np
import random
import shutil
import cv2


"0: surprise, 1: fear, 2: disgust, 3: happy 4: sad  5: angry 6: neutral 7:attempt"

class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False, ratio=1):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        self.file_paths = []
        # use raf-db aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        distribute = np.array(self.label)

        self.label_dis = [np.sum(distribute == 0), np.sum(distribute == 1), np.sum(distribute == 2),
                          np.sum(distribute == 3), \
                          np.sum(distribute == 4), np.sum(distribute == 5), np.sum(distribute == 6)]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (
        self.label_dis[0], self.label_dis[1], self.label_dis[2], self.label_dis[3], \
        self.label_dis[4], self.label_dis[5], self.label_dis[6]))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape=len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 3)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class FERPlus(data.Dataset):
    def __init__(self, path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        self.file_paths, self.label = [], []
        if self.phase == 'train':
            files = glob.glob(os.path.join(path, 'train/*/*.png'))
        else:
            files = glob.glob(os.path.join(path, 'test/*/*.png'))
            # files = glob.glob('fer+_wrong/*/*.jpg')
        for file in files:
            self.file_paths.append(file)
            self.label.append(int(file.split('/')[-2]))
        distribute = np.array(self.label)
        self.label_dis = [ np.sum(distribute == 0),  np.sum(distribute == 1),  np.sum(distribute == 2),  np.sum(distribute == 3),  \
                      np.sum(distribute == 4),  np.sum(distribute == 5),  np.sum(distribute == 6),  np.sum(distribute == 7),]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d, %d' % (self.label_dis[0], self.label_dis[1], self.label_dis[2],self.label_dis[3],\
                                                                          self.label_dis[4],self.label_dis[5],self.label_dis[6], self.label_dis[7]))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape = len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 3)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

class FER(data.Dataset):
    def __init__(self, path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise]
        self.file_paths, self.label = [], []
        if self.phase == 'train':
            files = glob.glob(os.path.join(path, 'train/*/*.jpg'))
            files += glob.glob(os.path.join(path, 'val/*/*.jpg'))
        else:
            files = glob.glob(os.path.join(path, 'test/*/*.jpg'))
        for file in files:
            self.file_paths.append(file)
            self.label.append(int(file.split('/')[-2]))
        distribute = np.array(self.label)
        self.label_dis = [ np.sum(distribute == 0),  np.sum(distribute == 1),  np.sum(distribute == 2),  np.sum(distribute == 3),  \
                      np.sum(distribute == 4),  np.sum(distribute == 5),  np.sum(distribute == 6)]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (self.label_dis[0], self.label_dis[1], self.label_dis[2],self.label_dis[3],\
                                                                          self.label_dis[4],self.label_dis[5],self.label_dis[6]))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape = len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx


class Oulu(data.Dataset):
    def __init__(self, path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        label_dict = { 0:6, 1:5,  2:2, 3:1, 4:3, 5:4, 6:0 }
        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        self.file_paths, self.label = [], []
        if self.phase == 'train':
            files = glob.glob(os.path.join(path, 'train/*/*/*.jpeg'))
        else:
            files = glob.glob(os.path.join(path, 'test/*/*/*.jpeg'))
        for file in files:
            self.file_paths.append(file)
            self.label.append(label_dict[int(file.split('/')[-2])])

        distribute = np.array(self.label)
        self.label_dis = [ np.sum(distribute == 0),  np.sum(distribute == 1),  np.sum(distribute == 2),  np.sum(distribute == 3),  \
                      np.sum(distribute == 4),  np.sum(distribute == 5),  np.sum(distribute == 6)]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (self.label_dis[0], self.label_dis[1], self.label_dis[2],self.label_dis[3],\
                                                                          self.label_dis[4],self.label_dis[5],self.label_dis[6]))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape = len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.1:
                index = random.randint(0, 3)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

class AFED(data.Dataset):
    def __init__(self,phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform
        label_dict = { 3:0, 6:1,  1:2, 5:3, 4:4, 9:5, 0:6 }

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop]
        self.files, self.labels, self.bboxs = [], [], []
        if self.phase == 'train':
            list_patition_label = pd.read_csv(
                './datasets/Asian_Facial_Expression/AsianMovie_0725_0730/list/train_list.txt', header=None,
                delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            for index in range(list_patition_label.shape[0]):
                if list_patition_label[index, -1] not in label_dict.keys():
                    continue
                bbox = list_patition_label[index, 1:5].astype(np.int)
                self.files.append(
                        './datasets/Asian_Facial_Expression/AsianMovie_0725_0730/images/' + list_patition_label[index, 0])
                self.labels.append(label_dict[list_patition_label[index, -1]])
                self.bboxs.append(bbox)
        else:
            list_patition_label = pd.read_csv(
                './datasets/Asian_Facial_Expression/AsianMovie_0725_0730/list/val_list.txt', header=None,
                delim_whitespace=True)
            list_patition_label = np.array(list_patition_label)
            for index in range(list_patition_label.shape[0]):
                if list_patition_label[index, -1] not in label_dict.keys():
                    continue
                bbox = list_patition_label[index, 1:5].astype(np.int)
                self.files.append(
                        './datasets/Asian_Facial_Expression/AsianMovie_0725_0730/images/' + list_patition_label[index, 0])
                self.labels.append(label_dict[list_patition_label[index, -1]])
                self.bboxs.append(bbox)

        distribute = np.array(self.labels)
        self.label_dis = [ np.sum(distribute == 0),  np.sum(distribute == 1),  np.sum(distribute == 2),  np.sum(distribute == 3),  \
                      np.sum(distribute == 4),  np.sum(distribute == 5),  np.sum(distribute == 6)]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (self.label_dis[0], self.label_dis[1], self.label_dis[2],self.label_dis[3],\
                                                                          self.label_dis[4],self.label_dis[5],self.label_dis[6]))

    def __len__(self):
        return len(self.files)

    def weight(self):
        return np.ones(shape = len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.files[idx]
        image = cv2.imread(path)
        box = self.bboxs[idx]
        image = image[box[1]:box[3], box[0]:box[2]]
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.labels[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx
#
# def load_data(path):
#     files = os.listdir(path)
#     for file in files:
#         label = file
#         file = os.path.join(path, file)
#         img_names = glob.glob(os.path.join(file, '*.jpg'))
#         np.random.shuffle(img_names)
#         num = len(img_names)
#         test_num = int(num * 0.1)
#         train_set = img_names[test_num:]
#         test_set = img_names[:test_num]
#         for img in train_set:
#             image = cv2.imread(img)
#             try:
#                 image = cv2.resize(image, (224, 224))
#                 img_ = img.split('/')
#                 target = os.path.join(path, 'train/'+'/'.join(img_[-2:]))
#                 target_path = os.path.join(path, 'train/'+'/'.join(img_[-2:-1]))
#                 if not os.path.exists(target_path):
#                     os.makedirs(target_path)
#                 shutil.copyfile(img, target)
#             except:
#                 print('error_image')
#         for img in test_set:
#             image = cv2.imread(img)
#             try:
#                 image = cv2.resize(image, (224, 224))
#                 img_ = img.split('/')
#                 target = os.path.join(path, 'test/'+'/'.join(img_[-2:]))
#                 target_path = os.path.join(path, 'test/'+'/'.join(img_[-2:-1]))
#                 if not os.path.exists(target_path):
#                     os.makedirs(target_path)
#                 shutil.copyfile(img, target)
#             except:
#                 print('error_image')


class AffectNet(data.Dataset):
    def __init__(self, path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        self.file_paths, self.label = [], []
        if self.phase == 'train':
            files = glob.glob(os.path.join(path, 'train/*/*.png'))
        else:
            files = glob.glob(os.path.join(path, 'test/*/*.png'))
            # files = glob.glob('fer+_wrong/*/*.jpg')
        for file in files:
            self.file_paths.append(file)
            self.label.append(int(file.split('/')[-2]))
        distribute = np.array(self.label)
        self.label_dis = [ np.sum(distribute == 0),  np.sum(distribute == 1),  np.sum(distribute == 2),  np.sum(distribute == 3),  \
                      np.sum(distribute == 4),  np.sum(distribute == 5),  np.sum(distribute == 6)]
        #print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (self.label_dis[0], self.label_dis[1], self.label_dis[2],self.label_dis[3],\
        #                                                                 self.label_dis[4],self.label_dis[5],self.label_dis[6]))

        if self.phase == 'train':
            files, labels = [], []
            mean = 5000#int(np.mean(self.label_dis))

            labels_index = [np.where(distribute == 0), np.where(distribute == 1), np.where(distribute == 2), np.where(distribute == 3),np.where(distribute == 4),\
                            np.where(distribute == 5), np.where(distribute == 6)]
            for idx, label_index in enumerate(labels_index):
                num = self.label_dis[idx]
                ls = list(map(int, label_index[0]))
                if num < mean:
                    lsl = random.sample(ls, mean-num)
                    files.extend([self.file_paths[i] for i in ls[:]])
                    labels.extend([self.label[i] for i in ls[:]])
                    files.extend([self.file_paths[i] for i in lsl[:]])
                    labels.extend([self.label[i] for i in lsl[:]])
                else:
                    lsl = random.sample(ls, mean)
                    files.extend([self.file_paths[i] for i in lsl[:]])
                    labels.extend([self.label[i] for i in lsl[:]])

#            files, labels = [], []
#            min_num = min(self.label_dis[0], self.label_dis[1], self.label_dis[2], self.label_dis[3], self.label_dis[4],
#                          self.label_dis[5], self.label_dis[6])
#            labels_index = [np.where(distribute == 0), np.where(distribute == 1), np.where(distribute == 2), np.where(distribute == 3),np.where(distribute == 4),\
#                            np.where(distribute == 5), np.where(distribute == 6)]
#            for label_index in labels_index:
#                ls = list(map(int, label_index[0]))
#                lsl = random.sample(ls, min_num)
#                files.extend([self.file_paths[i] for i in lsl[:]])
#                labels.extend([self.label[i] for i in lsl[:]])
            self.file_paths= files
            self.label= labels

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape = len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5   :
                index = random.randint(0, 3)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

class CAER(data.Dataset):
    def __init__(self, path, phase, transform=None, basic_aug=False):
        self.phase = phase
        self.transform = transform

        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        self.file_paths, self.label = [], []
        if self.phase == 'train':
            files = glob.glob(os.path.join(path, 'train/*/*.png'))
        else:
            files = glob.glob(os.path.join(path, 'test/*/*.png'))
        for file in files:
            self.file_paths.append(file)
            self.label.append(int(file.split('/')[-2]))

        distribute = np.array(self.label)
        self.label_dis = [ np.sum(distribute == 0),  np.sum(distribute == 1),  np.sum(distribute == 2),  np.sum(distribute == 3),  \
                      np.sum(distribute == 4),  np.sum(distribute == 5),  np.sum(distribute == 6)]
        print('The dataset distribute: %d, %d, %d, %d, %d, %d, %d' % (self.label_dis[0], self.label_dis[1], self.label_dis[2],self.label_dis[3],\
                                                                          self.label_dis[4],self.label_dis[5],self.label_dis[6]))

    def __len__(self):
        return len(self.file_paths)

    def weight(self):
        return np.ones(shape = len(self.label_dis)) / self.label_dis * 1000

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 3)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label, idx

class CelebA(data.Dataset):
    def __init__(self, path,  transform=None, basic_aug=False):
        self.transform = transform
        self.basic_aug = basic_aug
        self.aug_func = [util.flip_image, util.add_gaussian_noise, util.crop, util.rotation]
        self.file_paths = []
        files = glob.glob(os.path.join(path, '*.png'))
        for file in files:
            self.file_paths.append(file)
        print('dataset num:', str(len(self.file_paths)))

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        if self.basic_aug and random.uniform(0, 1) > 0.5:
            index = random.randint(0, 3)
            image = self.aug_func[index](image)
        if self.transform is not None:
            image = self.transform(image)

        return image, idx

if __name__ == '__main__':
    data = load_data(path = './datasets/ExpW')
