import random
import json
from torchvision import models
from torch import nn
from tqdm import tqdm
import torchattacks
import matplotlib.pyplot as plt
from arg_save import *
from hang_dataset import *

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


class Normalize(nn.Module):
    def __init__(self, mean, std):
        super(Normalize, self).__init__()
        self.register_buffer('mean', torch.Tensor(mean))
        self.register_buffer('std', torch.Tensor(std))

    def forward(self, input):
        # Broadcasting
        input = input / 255.0
        mean = self.mean.reshape(1, 3, 1, 1)
        std = self.std.reshape(1, 3, 1, 1)
        return (input - mean) / std


if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    class_idx = json.load(open("./imagenet_class_index.json"))
    idx2label = [class_idx[str(k)][1] for k in range(len(class_idx))]
    class2label = [class_idx[str(k)][0] for k in range(len(class_idx))]

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(), ])

    norm_layer = Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

    resnet_model = nn.Sequential(
        norm_layer,
        models.resnet50(pretrained=True)
    ).to(device)
    resnet_model = resnet_model.eval()
    eps = 8 / 255
    Adv = torchattacks.PGD(resnet_model, eps=eps)

    batch_size = 10
    data_dir = "./imagenet_5000"
    data_clean(data_dir)
    normal_data = image_folder_custom_label(root=data_dir, transform=transform, idx2label=class2label)
    normal_loader = torch.utils.data.DataLoader(normal_data, batch_size=batch_size, shuffle=True)
    normal_iter = iter(normal_loader)

    images, labels = normal_iter.next()
    setup_seed(10)

    os.environ["CUDA_VISIBLE_DEVICES"] = "6"
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # read model's args
    # model_args = {}

    # with open("./acc_record/res50_verify.txt", 'r') as f:
    #     for line in f:
    #         key, value = line.strip().split(':')
    #         model_args[key] = value

    parameters = {'model': 'res50',
              # 'model_TrainAcc': model_args['train_accuracy'],
              # 'model_TestAcc': model_args['test_accuracy'],
              'dataset': 'ImageNet-50k',
              'atk_method': 'PGD16',
              }
    res50 = torch.load("./models/res50-image50k.pth").to(device)
    for param in res50.parameters():
        param.requires_grad = False
    res50.eval()


