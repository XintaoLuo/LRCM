import numpy as np
import os

import torchvision
from torchvision import transforms
import torch
from sklearn.utils import shuffle
from sklearn.preprocessing import MinMaxScaler
import scipy.io
import matplotlib.pyplot as plt
import torchvision.datasets as dsets

class Dataset():
    def __init__(self, classifier_name, attack_name, dataset_name):
        self.adv_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/adv_result/'
        self.ori_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/origin_data/'
        self.classifier_name = classifier_name
        self.attack_name = attack_name
        self.dataset_name = dataset_name

    def load_origin_data(self):
        if self.dataset_name == 'cifar10':
            data_dir_path = self.ori_path + 'CIFAR10/'
            train_data, train_label = torch.load(data_dir_path + 'train_dataset.pt')
            test_data, test_label = torch.load(data_dir_path + 'test_dataset.pt')
        return train_data, train_label, test_data, test_label

    # 返回列表X(nxd) y最小值1(列表)
    def load_adv_data(self):
        data_dir_path = self.adv_path + self.classifier_name + '/'
        train_name = self.attack_name + '_' + self.dataset_name + '_train'
        test_name = self.attack_name + '_' + self.dataset_name + '_test'
        train_event = train_name + '.pt'
        test_event = test_name + '.pt'
        train_imgs = torch.load(data_dir_path + train_name + '/' + train_event, map_location=torch.device('cpu'))
        test_imgs = torch.load(data_dir_path + test_name + '/' + test_event, map_location=torch.device('cpu'))

        return train_imgs, test_imgs

    def normalize(self, x, min=0):
        # min_val = np.min(x)
        # max_val = np.max(x)
        # x = (x - min_val) / (max_val - min_val)
        # return x
        if min == 0:
            scaler = MinMaxScaler((0, 1))  # 列归一化
        else:  # min=-1
            scaler = MinMaxScaler((-1, 1))
        norm_x = scaler.fit_transform(x)
        return norm_x

    def Get_dataloaders(self, X_list, y_ture, batch_size, is_shuffle):
        from torch.utils.data import DataLoader, TensorDataset
        import torch
        # for i in range(len(X_list)):
        #     X_list[i] = torch.from_numpy(X_list[i]).float()
        X = TensorDataset(*X_list, y_ture)
        train_loader = DataLoader(X, batch_size=batch_size, shuffle=is_shuffle)
        return train_loader

    def load_net_para(self):
        args = {
            'constant': {
                'data_name': 'CIFAR10',
                'encoder_bias': True,
                'epochs': 15,
                'Batch_size': 128,
                'lr': 5e-3,
                'begin_tuning_times': 0,  # 1-49(两重调参)
                'show_result_epoch': 2,
            },
            'quan_bit' : 8,
            'quan_alpha' : 0.2,
            'para_entropy' : 0.01,
            'conv_mindim': 16,
            # 'para_normal_6': 1,
            # 'para_rec': 1,
            # 'para_kl': 1,
            # 'para_binary': 1e-2,
            'acc': {
                'adv_acc': 0.0,
                'pure_acc': 0.0,
            },
            # 'atk_name': 'FGSM8',
            'ori_train_psnr_history': [],
            'adv_train_psnr_history': [],
            'Use_cuda': True,
            'model':{
                'model_name': 'resnet18',
            },
            'verify':{
                'adv_train_psnr': 0,
                'ori_train_psnr': 0,
            },
            'other': {
                'Is_shuffle': False,

                # fig
                'plot_first': False,
                'save_result': True
            }
        }
        # if self.dataset_name == 'cifar10':
        #     args['constant']['Batch_size'] = 128

        return args


def save_data(data_name, dir_name):
    if data_name == 'cifar10':
        train_dataset = torchvision.datasets.CIFAR10(
            root='CIFAR10',
            train=True,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                               std=[0.2023, 0.1994, 0.2010])]),
            download=False
        )
        test_dataset = torchvision.datasets.CIFAR10(
            root='CIFAR10',
            train=False,
            transform=transforms.Compose([transforms.ToTensor(),
                                          transforms.Normalize(mean=[0.4914, 0.4822, 0.4465],
                                                               std=[0.2023, 0.1994, 0.2010])]),
            download=False
        )
    else:
        train_data = None
        test_data = None
        label = None

    # train_data = torch.tensor(train_dataset.data).permute(0, 3, 1, 2)
    train_data_list = []
    test_data_list = []
    for i in range(len(train_dataset)):
        train_data_i, _ = train_dataset[i]
        train_data_list.append(torch.tensor(train_data_i))
    for i in range(len(test_dataset)):
        test_data_i, _ = test_dataset[i]
        test_data_list.append(torch.tensor(test_data_i))
    train_data = torch.stack(train_data_list)
    train_label = torch.tensor(train_dataset.targets)
    test_data = torch.stack(test_data_list)
    test_label = torch.tensor(test_dataset.targets)
    print(train_data)
    save_path = os.path.dirname(os.path.realpath(__file__)) + '/' + 'origin_data' + '/' + dir_name + '/'
    torch.save([train_data, train_label], save_path + 'train_dataset.pt')
    torch.save([test_data, test_label], save_path + 'test_dataset.pt')
    print('save {} over!!!'.format(data_name))


# imagenetdataset
def imshow(img, title):
    npimg = img.numpy()
    fig = plt.figure(figsize=(5, 15))
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.title(title)
    plt.show()


def image_folder_custom_label(root, transform, idx2label):
    # custom_label
    # type : List
    # index -> label
    # ex) ['tench', 'goldfish', 'great_white_shark', 'tiger_shark']

    old_data = dsets.ImageFolder(root=root, transform=transform)
    old_classes = old_data.classes

    label2idx = {}

    for i, item in enumerate(idx2label):
        label2idx[item] = i

    new_data = dsets.ImageFolder(root=root, transform=transform,
                                 target_transform=lambda x: idx2label.index(old_classes[x]))
    new_data.classes = idx2label
    new_data.class_to_idx = label2idx

    return new_data


def create_dir(dir, print_flag=False):
    if not os.path.exists(dir):
        os.mkdir(dir)
        if print_flag:
            print("Create dir {} successfully!".format(dir))
    elif print_flag:
        print("Directory {} is already existed. ".format(dir))


def data_clean(data_dir):
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isfile(class_path):
            os.remove(class_path)
        for img_name in os.listdir(class_path):
            img_path = os.path.join(class_path, img_name)
            if not img_name.endswith(".png"):
                os.remove(img_path)

def calculate_entropy(image_data):
    # 计算图像数据的信息熵
    histogram = np.histogram(image_data, bins=256, range=(0, 1))[0]
    histogram = histogram / histogram.sum()  # 归一化直方图
    entropy = -np.sum(histogram * np.log2(histogram + np.finfo(float).eps))
    return entropy


# if __name__ == '__main__':
#     save_data('cifar10', 'CIFAR10')
