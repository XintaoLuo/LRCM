import os

import matplotlib.pyplot as plt
import torch
import torchvision
from sklearn.preprocessing import MinMaxScaler
from torchvision import transforms

from plot_fig import *


class Dataset():
    def __init__(self, classifier_name, attack_name, dataset_name):
        self.adv_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/adv_result/'
        self.ori_path = os.path.abspath(os.path.dirname(os.path.realpath(__file__))) + '/origin_data/'
        self.classifier_name = classifier_name
        self.attack_name = attack_name
        if 'cifar100' in dataset_name:
            self.dataset_name = 'cifar100'
        else:
            self.dataset_name = 'cifar10'

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
                'epochs': 500,
                'Batch_size': 100,
                'lr': 0.005,
                'begin_tuning_times': 0,  # 1-49(两重调参)
                'show_result_epoch': 50,
            },
            'conv_mindim': 16,
            'para_normal_6': 1,
            'para_rec': 1,
            'para_kl': 1,
            'Use_cuda': False,
            'other': {
                # fig
                'plot_first': True,
                'save_result': True
            }
        }
        if self.dataset_name == 'cifar10':
            args['constant']['Batch_size'] = 5000

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


def save_args(dir_name, txt_name, dict):
    project_path = os.path.abspath(os.path.join(os.path.dirname(os.path.realpath(__file__)), './'))
    save_path = project_path + '/' + dir_name
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    with open(save_path + '/' + txt_name, "w") as f:
        for k, v in dict.items():
            f.write(k + ':' + str(v))
            f.write('\n')

    print('file was writen')


def args_dict(dict_initial):
    lr = 0.005
    batch_size = 2000
    epoch = 10
    args = {
        'lr': lr,
        'batch_size': batch_size,
        'epoch': epoch,
    }
    args.update(dict_initial)
    return args


def save_adv_images(adv_image, ori_image, args):
    dir = './adv_result/' + args['model'] + '/' + args['atk_method'] + '_' + args['dataset'] + '/' + args[
        'atk_method'] + '_' + args['dataset']
    if not os.path.exists('./adv_result/' + args['model'] + '/' + args['atk_method'] + '_' + args['dataset']):
        os.makedirs('./adv_result/' + args['model'] + '/' + args['atk_method'] + '_' + args['dataset'])
    adv_images_total = torch.cat(adv_image, dim=0)
    torch.save(adv_images_total, dir + '.pt')
    save_args('adv_result/' + args['model'] + '/' + args['atk_method'] + '_' + args['dataset'],
              args['atk_method'] + '_' + args['dataset'] + '.txt', args)
    if args['dataset'][:5] == 'mnist':
        plot_advMNIST(adv_image, adv_image[1], 5, args['atk_method'])
    else:
        plot_result([adv_image[0][:5], ori_image[:5]], [args['atk_method'], 'ori_images'], fig_saved_path=dir + '.png')
        # plot_result([adv_image[0][:5], ori_image[:5]], [args['atk_method'], 'ori_images'])
    print('saved adv_images')



#
# if __name__ == '__main__':
#     dict_initial = {
#         'model': "res18",
#         'dataset': "CIFAR10",
#         'Normalize': "mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010]",
#     }
#     args = args_dict(dict_initial)
#     save_args("acc_record", "res18_verify.txt", args)
