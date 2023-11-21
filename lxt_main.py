
import warnings

from matplotlib import pyplot as plt
from scipy import ndimage

warnings.filterwarnings("ignore")
from sklearn.cluster import KMeans
from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()
import numpy as np
import torch
import scipy.io as scio

import random
from hang_dataset import *
from hang_model import *
from torch import optim
from lxt_trainer import Trainer, plot_result


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


setup_seed(5)

import os
os.environ['CUDA_VISIBLE_DEVICES'] = '2'
device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")


def main(attatck_name, dataset_name='cifar10'):
    data = Dataset('res18', attatck_name, dataset_name)
    adv_train, adv_test = data.load_adv_data()
    ori_train, train_label, ori_test, test_label = data.load_origin_data()

    train_num = len(ori_train)
    test_num = len(ori_test)
    args = data.load_net_para()
    # if args['other']['plot_first']:
    #     plot_result([adv_train[:5], ori_train[:5], adv_test[:5], ori_test[:5]],
    #                 ['adv_train', 'ori_train', 'adv_test', 'ori_test'])
    use_cuda = torch.cuda.is_available()
    print("cuda is available?")
    print(use_cuda)
    args['Use_cuda'] = use_cuda
    args['atk_name'] = attatck_name
    args['constant']['freeze_epoch'] = 0
    para_list = [4,6,8,10,12]
    args['para_binary'] = 0.001
    TEST = False
    if TEST == False:
        train_loader = data.Get_dataloaders([ori_train, adv_train], train_label,
                                            args['constant']['Batch_size'], args['other']['Is_shuffle'])
        test_loader = data.Get_dataloaders([ori_test, adv_test], test_label, test_num, is_shuffle=False)
        tuning_times = 0
        for args['quan_bit'] in para_list:
            # for args['para_rec'] in para_list:
            # args['para_rec'] = 1

            tuning_times += 1
            if tuning_times < args['constant']['begin_tuning_times']:
                continue
            # Build a model

            model = QuantizeUNet(args)
            optimizer = optim.Adam(model.parameters(), lr=args['constant']['lr'])
            if use_cuda:
                model.cuda()
            # print(args)
            # print(model)

            trainer = Trainer(model, optimizer, train_loader, test_loader, args)
            trainer.easy_train()

            model_name = './lxt_result/QuantizeUNet_' + attatck_name + '.pt'
            torch.save(trainer.model.state_dict(), model_name)
            print('Model saved as', model_name)

        # torch.save(trainer.model.state_dict(), './models/' + model_name)
        # print('save model?')


if __name__ == '__main__':
    attatck_name_list = ['CW']
    for attatck_name in attatck_name_list:
        main(attatck_name)
