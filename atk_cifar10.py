import random

from tqdm import tqdm
import torchattacks

from arg_save import *


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
setup_seed(10)

os.environ["CUDA_VISIBLE_DEVICES"] = "6"
device = torch.device("cuda:6" if torch.cuda.is_available() else "cpu")
# read model's args
model_args = {}

with open("./acc_record/res18_verify.txt", 'r') as f:
    for line in f:
        key, value = line.strip().split(':')
        model_args[key] = value
        

parameters = {'model': 'res18',
              'model_TrainAcc': model_args['train_accuracy'],
              'model_TestAcc': model_args['test_accuracy'],
              'dataset': 'cifar10_train',
              'atk_method': 'PGD16',
}
res18 = torch.load("./models/res18-cifar10.pth").to(device)
for param in res18.parameters():
    param.requires_grad = False


batch_size = 100
Data = Dataset(parameters['model'], parameters['atk_method'], parameters['dataset'][:5])
ori_train, train_label, ori_test, test_label = Data.load_origin_data()
train_loader = Data.Get_dataloaders([ori_train], train_label, batch_size, is_shuffle=False)
test_loader = Data.Get_dataloaders([ori_test], test_label, batch_size, is_shuffle=False)
#
# loss, pgd10_acc = attack.eval_robust(res18, train_loader, epsilon=8/255)
# print('FGSM10 Test Accuracy: {:.2f}%'.format(100. * pgd10_acc))


atk = torchattacks.PGD(res18, eps=16/255)
# If, images are normalized:
atk.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])


for e in range(2):

    adv_true = 0  # under adv, still correct
    total = 0
    pbar1 = tqdm(train_loader)
    adv_train_list = []
    for step, (x, y) in enumerate(pbar1):

        x = x.cuda()
        y = y.cuda()
        adv_images = atk(x, y).detach()
        adv_train_list.append(adv_images)

        outputs = res18(adv_images)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        adv_true += (predicted == y).sum().item()
        pbar1.set_postfix(Epoch=e, SucRate=1 - (adv_true / total))
        if step == 70 and e == 0:
            break


    parameters['success_rate'] = 1 - (adv_true / total)
    print("attack_rate = '{:.4%}'".format(1 - (adv_true / total)))
    # save_adv_images(adv_train_list, ori_train, parameters)

for e in range(2):

    adv_true = 0  # under adv, still correct
    total = 0
    pbar2 = tqdm(test_loader)
    adv_train_list = []
    for step, (x, y) in enumerate(pbar2):

        x = x.cuda()
        y = y.cuda()
        adv_images = atk(x, y).detach()
        adv_train_list.append(adv_images)

        outputs = res18(adv_images)
        _, predicted = torch.max(outputs, 1)
        total += y.size(0)
        adv_true += (predicted == y).sum().item()
        pbar2.set_postfix(Epoch=e, SucRate=1 - (adv_true / total))
        if step == 30 and e == 0:
            break

    parameters['dataset'] = 'cifar10_test'
    parameters['success_rate'] = 1 - (adv_true / total)
    print("attack_rate = '{:.4%}'".format(1 - (adv_true / total)))
    # save_adv_images(adv_train_list, ori_train, parameters)


