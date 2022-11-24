

import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import random
import numpy as np
import torch
import cv2
import timm
import pandas as pd
import copy
import matplotlib.pyplot as plt
import torchvision.transforms as transform
from sklearn import model_selection
from torch.utils.data import Dataset
import torch.nn as nn
import torch.optim as optim

from torch.optim.lr_scheduler import StepLR, CosineAnnealingWarmRestarts, CosineAnnealingLR
from torch.utils.data import DataLoader
import models.create_models as create
from tqdm import tqdm


from dataset import CustomDataset, MultiviewImgDataset, SingleimgDataset
from models.FLoss import FocalLoss

def seed_everything(seed):
    """
    Seeds basic parameters for reproductibility of results

    Arguments:
        seed {int} -- Number of the seed
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main(model_name,N_EPOCHS=100,LR = 0.0001,depth=12,head=9):
    # model = ViT(num_classes=5, pretrained=True)
    # model = Deit(num_classes=5, pretrained=True)
    # model = ResNet50(num_classes=5 , heads=4)
    print(model_name)
    model = create.my_MVCINN(
        pretrained=True,
        num_classes=5,
        pre_Path = 'weights/final_0.8010.pth',
        depth=depth,
        num_heads=head,
        # embed_dim=768,
        drop_rate=0.2,
        drop_path_rate=0.2,

    )



    optimizer = optim.Adam(model.parameters(), lr=LR)

    scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=4, T_mult=2)
    # scheduler = CosineAnnealingLR(optimizer,T_max=5)
    # criterion = nn.CrossEntropyLoss()
    criterion = FocalLoss(gamma=2)
    train_loss = []
    train_acc = []
    valid_loss = []
    valid_acc = []
    inter_val = 2

    best_model = copy.deepcopy(model)

    last_model = model
    model.to(device)

    best_acc = 0
    best_test = 0
    for epoch in range(N_EPOCHS):

        train_epoch_loss = 0.0
        train_epoch_acc = 0.0

        model.train()
        train_bar = tqdm(train_loader)

        for i, (img, label) in enumerate(train_bar):
            img = img.to(device, non_blocking=True)
            label = label.to(device, non_blocking=True)

            B, V, C, H, W = img.size()
            
            # mixup
            alpha=0.2
            lam = np.random.beta(alpha,alpha)
            index = torch.randperm(B).cuda()
            imgs_mix = lam*img + (1-lam)*img[index]
            label_a,label_b = label,label[index]
            

            imgs_mix = imgs_mix.view(-1, C, H, W)

            # b,v = label.shape
            # label=label.view(b*v)
            optimizer.zero_grad()
            output,_ = model(imgs_mix)
            output = output[1]+output[0]

            # print(label.shape,output.shape)
            loss = lam*criterion(output, label_a) + (1-lam)*criterion(output, label_b)



            loss.backward()
            train_epoch_acc += (output.argmax(dim=1) == label).sum()
            train_epoch_loss += loss.item()


            scheduler.step(epoch + i / len(train_loader))
            optimizer.step()



        train_loss_mean = train_epoch_loss / len(train_loader)
        train_acc_mean = train_epoch_acc / (len(train_dataset) * NUM_VIEW)


        train_loss.append(train_loss_mean)
        train_acc.append(train_acc_mean.cpu())

        print('{} train loss: {:.3f} train acc: {:.3f}  lr:{}'.format(epoch,train_loss_mean,
                                                                        train_acc_mean,
                                                                        optimizer.param_groups[-1]['lr']))
        if (epoch + 1) % inter_val == 0:
            val_acc_mean,  val_loss_mean = val_model(model, test_loader, criterion,device)

            if val_acc_mean > best_acc:
                model.cpu()
                best_model = copy.deepcopy(model.state_dict())
                model.to(device)
                best_acc = val_acc_mean

            valid_loss.append(val_loss_mean)
            valid_acc.append(val_acc_mean.cpu())


        # test_acc_mean = testModel(model,test_loader,len(test_dataset),device)
        # if test_acc_mean>best_test:
        #     best_test = test_acc_mean

    print("best val acc:", best_acc)
    if best_test != 0: print("best test acc:", best_test)
    torch.save(best_model, os.path.join(SAVE_PT_DIR,'{}-{:.4f}.pth'.format(model_name,best_acc)))

    # torch.save(model, os.path.join(SAVE_PT_DIR,'last2.pt'))
    print("model saved at weights")




def val_model(model, valid_loader, criterion, device):
    valid_epoch_loss = 0.0
    valid_epoch_acc = 0.0

    model.eval()
    valid_bar = tqdm(valid_loader)
    for i, (img, label) in enumerate(valid_bar):
        img = img.to(device)
        label = label.to(device)

        B, V, C, H, W = img.size()
        img = img.view(-1, C, H, W)
        img = img.to(device, non_blocking=True)

        # b,v = label.shape
        # label=label.view(b*v)

        with torch.no_grad():
            output,_ = model(img)
        output = output[1]+output[0]
        
        
        loss = criterion(output, label)
        valid_epoch_loss += loss.item()
        valid_epoch_acc += (output.argmax(dim=1) == label).sum()


    val_acc_mean = valid_epoch_acc / (len(valid_dataset) * NUM_VIEW)
    val_loss_mean = valid_epoch_loss / len(valid_loader)

    
    print('valid loss: {:.3f} valid acc: {:.3f}'.format(val_loss_mean,val_acc_mean))
    return val_acc_mean, val_loss_mean


if __name__ == '__main__':
    seed_everything(1001)
    # general global variables
    DATA_PATH = "../EYData_BaseEye_newdata/"
    TRAIN_PATH = "../EYData_BaseEye_newdata/train/rgb"
    TEST_PATH = "../EYData_BaseEye_newdata/test/rgb"
    SAVE_IMG_DIR = 'imgs'
    SAVE_PT_DIR = 'weights'
    NUM_VIEW = 1
    IMAGE_SIZE = 224
    LR = 0.00001
    N_EPOCHS = 50
    DEPTH = 12
    HEAD = 9
    BATCH_SIZE = 8
    train_csv_path = os.path.join(DATA_PATH, 'train_rgb_label_newname.csv')
    assert os.path.exists(train_csv_path), '{} path is not exists...'.format(train_csv_path)
    test_csv_path = os.path.join(DATA_PATH, 'test_rgb_label_newname.csv')
    test_df = pd.read_csv(test_csv_path)

    all_data = pd.read_csv(train_csv_path)
    all_data.head()


    transform_train = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.RandomHorizontalFlip(p=0.3),
        transform.RandomVerticalFlip(p=0.3),
        transform.RandomResizedCrop(IMAGE_SIZE),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
    ])

    transform_valid = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])
    transform_test = transform.Compose([
        transform.ToPILImage(),
        transform.Resize((224, 224)),
        transform.ToTensor(),
        # transform.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    ])

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


    train_dataset = MultiviewImgDataset(TRAIN_PATH, all_data, transform=transform_train)

    valid_dataset = MultiviewImgDataset(TEST_PATH, test_df, transform=transform_test)
    test_dataset = MultiviewImgDataset(TEST_PATH, test_df, transform=transform_test)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=8, pin_memory=False)
    valid_loader = DataLoader(valid_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=8, pin_memory=False)
    print(len(train_dataset), len(valid_dataset), len(test_dataset))
    main(model_name=f'main_{LR}_{N_EPOCHS}_d{DEPTH}_h{HEAD}',N_EPOCHS = N_EPOCHS,LR = LR,depth=DEPTH,head=HEAD)