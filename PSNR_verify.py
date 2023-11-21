from hang_dataset import *
from hang_model import *
import torch
import random
from sklearn.metrics import mean_squared_error
from math import log10
import numpy as np

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(5)

os.environ['CUDA_VISIBLE_DEVICES'] = '6'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model(model_path, args):
    model = FreezUNet(args)  # 创建模型实例，其中 args 包含模型的参数配置
    model.load_state_dict(torch.load(model_path))
    model.eval().cuda() # 设置模型为评估模式
    return model


def PSNR_Entro(image1, image2):
    # 初始化变量来保存 PSNR 和样本数

    total_psnr_batch = 0
    total_psnr_all = 0
    num_samples = 0
    # 遍历每个样本
    for i in range(len(image1)):
        sample_adv = image1[i].reshape(-1)
        sample_ori = image2[i].reshape(-1)

        mse = mean_squared_error(sample_ori, sample_adv)
        # if mse < 1e-6:
        #     mse = 1e-6

        # 计算单个样本的 PSNR
        psnr = 10 * log10((2.75 ** 2) / (mse+1e-8))

        # 累加 PSNR 和样本数
        total_psnr_batch += psnr
        total_psnr_all += psnr
        num_samples += 1
    # 计算整个批次的平均 PSNR
    average_psnr_batch = total_psnr_batch / num_samples

    # 计算所有数据的平均 PSNR
    average_psnr_all = total_psnr_all / num_samples

    # for i in range(10):
    #     sample_adv = image1[i].reshape(-1)  # 将图像重塑为一维数组
    #     sample_ori = image2[i].reshape(-1)
    #     mse = mean_squared_error(sample_ori, sample_adv)
    #
    #     psnr = 10 * log10(1 / mse)

    # 计算信息熵
    entropy = calculate_entropy(image2)

    return average_psnr_all, entropy

# 计算信息熵
def calculate_entropy(image_data):
    batch_entropy = 0
    num_samples = len(image_data)

    for i in range(len(image_data)):
        histogram = np.histogram(image_data[i], bins=256, range=(0, 1))[0]
        histogram = histogram / (histogram.sum()+1e-9)
        entropy = -np.sum(histogram * np.log2(histogram + np.finfo(float).eps))
        batch_entropy += entropy

    # 计算整个批次的平均熵
    average_entropy_batch = batch_entropy / num_samples

    return average_entropy_batch


def main():
    atk = 'CW'
    data = Dataset('res18', atk, 'cifar10')
    adv_train, adv_test = data.load_adv_data()
    ori_train, train_label, ori_test, test_label = data.load_origin_data()
    args = data.load_net_para()
    train_loader = data.Get_dataloaders([ori_train, adv_train], train_label,
                                        args['constant']['Batch_size'], args['other']['Is_shuffle'])
    # test_loader = data.Get_dataloaders([ori_test, adv_test], test_label, test_num, is_shuffle=False)
    UNet = load_model('./lxt_result/FreezUNet_'+atk+'.pt', args)
    for step, (ori_x, adv_x, y) in enumerate(train_loader):
        with torch.no_grad():
            adv_train_tensor = adv_x.cuda()  # 将 adv_train 转换为 Tensor 并移动到设备
            ori_tensor = ori_x.cuda()
            reconstructed_images = UNet(adv_train_tensor).cpu()
        adv_train_psnr, rec_entropy = PSNR_Entro(ori_x, reconstructed_images)
        unet_reconstructed_psnr, ori_entropy = PSNR_Entro(adv_x, ori_x)
        adv_entropy = calculate_entropy(adv_x)
        print("Step:", step)
        print("PSNR between ori_train and rec_train:", adv_train_psnr)

        print("PSNR between UNet adv_train and ori_train:", unet_reconstructed_psnr)
        print("Information Entropy of rec_train:", rec_entropy)
        print("Information Entropy of origin:", ori_entropy)
        print("Information Entropy of adv:", adv_entropy)

if __name__ == '__main__':
    main()
    # 使用 loaded_model 进行预测或其他操作
