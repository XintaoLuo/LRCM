import os

import numpy as np
import torch
from matplotlib import pyplot as plt
from PSNR_verify import *
import torchattacks

class Trainer():
    def __init__(self, model, optimizer, train_loader, test_loader, args):
        self.model = model
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.args = args

        self.res18 = torch.load("./models/res18-cifar10.pth").cuda()

        # self.PGD = torchattacks.PGD(self.res18, eps=16/255)
        # self.PGD.set_normalization_used(mean=[0.4914, 0.4822, 0.4465], std=[0.2023, 0.1994, 0.2010])
    def easy_train(self):
        dir_path = create_result_dir(self.args) if self.args['other']['save_result'] else None
        highest_pure_rate = 0
        for e in range(self.args['constant']['epochs']):

            if e == self.args['constant']['freeze_epoch']:
                self.model.freeze_encoder = True
                print('Freeze Encoder in epoch {}'.format(e))
            batch_num = 0
            clean_true = 0
            recon_true = 0
            pure_true = 0
            highest_epoch = 0
            adv_true = 0
            for step, (ori_x, adv_x, y) in enumerate(self.train_loader):
                if self.args['Use_cuda']:
                    ori_x = ori_x.clone().to('cuda')

                self.optimizer.zero_grad()
                recon_x = self.model(ori_x)
                # loss_fn = F.cross_entropy
                # pr1 = self.res18(ori_x).cuda()
                # loss_l = loss_fn(pr1, y.cuda()).cuda()
                # 计算原始图像和压缩后的图像的信息熵
                entropy_original = image_entropy(ori_x)
                entropy_compressed = image_entropy(recon_x)
                # 计算熵最小化损失
                entropy_loss = torch.abs(entropy_original - entropy_compressed)
                # entropy_loss = entropy_compressed
                # mse_loss, kl_loss = self.model.loss_vae_mse(ori_x, recon_x, mu, logvar)
                # loss_total = self.args['para_rec']*mse_loss + self.args['para_kl']*kl_loss
                if self.args['model']['denoise'] != 'AENet':

                    loss_total = self.args['para_entropy']*entropy_loss \
                                 + self.model.loss_mse(ori_x, recon_x)      \
                                    # + loss_l*0.001



                else:
                    loss_total = self.model.loss_mse(ori_x, recon_x)

                loss_total.backward()
                self.optimizer.step()
                # adv_x = self.PGD(ori_x.cuda(), y.cuda()).detach()
                with torch.no_grad():
                    batch_num += self.args['constant']['Batch_size']

                    if e == 0:
                        pred = torch.topk(self.res18(ori_x.cuda()).detach(), 1)[1].squeeze(1).cpu()
                        clean_true += pred.eq(y.cpu()).sum().numpy()
                        clean_rate = (clean_true / batch_num)

                        pred1 = torch.topk(self.res18(adv_x.cuda()).detach(), 1)[1].squeeze(1).cpu()
                        adv_true += pred1.eq(y.cpu()).sum().numpy()
                        adv_rate = (adv_true / batch_num)

                    rec_x = recon_x.clone().detach().cuda()
                    pred2 = torch.topk(self.res18(rec_x).detach(), 1)[1].squeeze(1).cpu()
                    recon_true += pred2.eq(y.cpu()).sum().numpy()
                    recon_rate = (recon_true / batch_num)

                    pure_x = self.model(adv_x.cuda())
                    pred3 = torch.topk(self.res18(pure_x.cuda()).detach(), 1)[1].squeeze(1).cpu()
                    pure_true += pred3.eq(y.cpu()).sum().numpy()
                    pure_rate = (pure_true / batch_num)

                if pure_rate > highest_pure_rate:
                    highest_epoch = e
                    highest_pure_rate = pure_rate
                if step % 10 == 0:
                    # print('Epoch :', e, '|', 'train_loss:%.5f' % loss_total.data, 'kld_loss:%.5f' % kl_loss.data)
                    print('Epoch :', e, '|', 'total_loss:%.5f' % loss_total.data,

                          # 'clean_rate:%.5f' % clean_rate,
                            'recon_rate:%.5f' % recon_rate,
                          'pure_rate:%.5f' % pure_rate)


                    adv_train_psnr, _ = PSNR_Entro(adv_x.cpu(), pure_x.cpu())
                    ori_train_psnr, rec_entropy = PSNR_Entro(ori_x.cpu(), pure_x.cpu())
                    self.args['verify']['adv_train_psnr'] = adv_train_psnr
                    self.args['verify']['ori_train_psnr'] = ori_train_psnr

                    self.args['acc']['clean_acc'] = clean_rate
                    self.args['acc']['adv_acc'] = adv_rate
                    self.args['acc']['highest_epoch'] = highest_epoch
                    self.args['acc']['rec_acc'] = recon_rate
                    self.args['acc']['pure_acc_best'] = highest_pure_rate
                    self.args['acc']['pure_acc_final'] = pure_rate

            if e % self.args['constant']['show_result_epoch'] == 0 or e == self.args['constant']['epochs'] - 1:
                with torch.no_grad():
                    self.model.training = False
                    ori_t, adv_t, y_t = self.train_loader.dataset[23:32]
                    if self.args['Use_cuda']:
                        ori_t = ori_t.clone().to('cuda')
                        adv_t = adv_t.clone().to('cuda')
                    # recon_test, _, _ = self.model(ori_t)
                    recon_ori = self.model(ori_t)
                    recon_adv = self.model(adv_t)
                    self.args['ori_train_psnr_history'].append(self.args['verify']['ori_train_psnr'])
                    self.args['adv_train_psnr_history'].append(self.args['verify']['adv_train_psnr'])

                    self.args['verify']['rec_entropy'] = rec_entropy
                    self.model.training = True



                fig_path = dir_path + '/' + str(e) + '.png' if self.args['other']['save_result'] else None
                # plot_result([ori_t.detach().cpu().data, recon_ori.detach().cpu().data,
                #              adv_t.detach().cpu().data, recon_adv.detach().cpu().data],
                #             ['ori_data','recon_ori', 'adv_data', 'recon_adv epoch:{}'.format(e)],
                #             self.args, txt_path=dir_path + '/' + str(e), fig_saved_path=fig_path)
                plot_result([ori_t.detach().cpu().data,
                             adv_t.detach().cpu().data, recon_adv.detach().cpu().data],
                            ['Origin Images', 'Adversarial Images', 'Reconstructed Images'],
                            self.args, txt_path=dir_path + '/' + str(e), fig_saved_path=fig_path)
                # epsfile
                # fig_path = dir_path + '/' + str(e) + '.eps' if self.args['other']['save_result'] else None
                # plot_result([ori_t.detach().cpu().data,
                #              adv_t.detach().cpu().data, recon_adv.detach().cpu().data],
                #             ['Origin Images', 'Adversarial Images', 'Reconstructed Images'],
                #             self.args, txt_path=dir_path + '/' + str(e), fig_saved_path=fig_path)


def create_result_dir(args):
    di = 0
    while (1):
        dir_path = 'lxt_result/' + args['constant']['data_name'] + 'Quan_bit/' + str(di)
        if not os.path.exists(dir_path):
            os.makedirs(dir_path)
            break
        else:
            di += 1

    return dir_path


def plot_result(imgdata, title, args=None, txt_path=None, fig_saved_path=None):
    if type(imgdata) is not list:
        n = len(imgdata)
        imgdata = imgdata.permute(0, 2, 3, 1).cpu().detach().numpy()
        imgdata = (imgdata * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
        imgdata = np.clip(imgdata, 0, 255)
        imgdata = imgdata.astype('uint8')
        fig, axes = plt.subplots(1, n, figsize=(1 * n, 15))

        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=0.5, hspace=1)
        for i in range(n):
            im_result = imgdata[i]
            axes[i].set_xticks([])
            axes[i].set_yticks([])
            axes[i].imshow(im_result)
        plt.title(title, fontsize=15, fontweight='bold')
        # if fig_saved_path is not None:
        #
        #     fig.savefig(fig_saved_path)
        #     print('fig have saved!!! path is {}'.format(fig_saved_path))
        # else:
        #     fig.show()
    else:
        dtype = len(imgdata)
        num = len(imgdata[0])
        fig, axes = plt.subplots(dtype, num, figsize=(dtype * num, 3 * dtype))
        plt.subplots_adjust(left=None, bottom=None, right=None, top=None, wspace=None, hspace=None)
        for d in range(dtype):
            axes[d][0].set_title(title[d], fontsize=6 * dtype, fontweight='bold')
            temp_data = imgdata[d].permute(0, 2, 3, 1).cpu().detach().numpy()
            temp_data = (temp_data * [0.2023, 0.1994, 0.2010] + [0.4914, 0.4822, 0.4465]) * 255
            temp_data = np.clip(temp_data, 0, 255)
            temp_data = temp_data.astype('uint8')
            for n in range(num):
                im_result = temp_data[n]
                axes[d][n].set_xticks([])
                axes[d][n].set_yticks([])
                axes[d][n].imshow(im_result)

    if args is not None:
        str_args = ''
        for key, value in args.items():
            if key == 'constant':
                str_args += (str(args[key]) + '\n')
            elif key == 'other':
                continue
            else:
                str_args += (str(key) + ':' + str(value) + ' ')

        # fig.suptitle(str_args, fontsize=15)
        # fig.suptitle("Comparison with Reconstructed Images and Adversarial Images", fontsize=15)

    if txt_path is not None:
        with open(txt_path + 'args.txt', "w") as f:
            for k, v in args.items():
                f.write(k + ':' + str(v))
                f.write('\n')
        print('write args finished')
    else:
        for k, v in args.items():
            print(k + ':' + str(v))
            print('\n')

    if fig_saved_path is not None:

        fig.savefig(fig_saved_path)
        print('fig have saved!!! path is {}'.format(fig_saved_path))
    else:
        fig.show()



def image_entropy(image):

    image = image.clone().detach().requires_grad_(True)


    image = (image - torch.min(image)) / (torch.max(image) - torch.min(image))


    prob_dist = torch.histc(image, bins=256, min=0, max=1) / image.numel()

    entropy = -torch.sum(prob_dist * torch.log2(prob_dist + 1e-10))

    return entropy



#
# def compute_psnr(img1, img2):
#     mse = torch.mean((img1 - img2) ** 2)
#     if mse == 0:
#         return 100  # PSNR is infinity if images are identical
#     max_pixel = 255.0
#     psnr = 20 * torch.log10(max_pixel / torch.sqrt(mse))
#     return psnr
#
# def test_model_psnr(model, test_loader):
#     model.eval()
#     psnr_values = []
#
#     for data, target in test_loader:
#         if torch.cuda.is_available():
#             data, target = data.cuda(), target.cuda()
#
#         with torch.no_grad():
#             output = model(data)
#
#         # Calculate PSNR for each image in the batch
#         for i in range(data.size(0)):
#             psnr = compute_psnr(data[i], output[i])
#             psnr_values.append(psnr.item())
#
#     # Compute the average PSNR over the entire test set
#     avg_psnr = np.mean(psnr_values)
#     return avg_psnr