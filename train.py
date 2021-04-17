import pandas as pd
import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from tqdm.notebook import tqdm

from sklearn.metrics import f1_score, roc_auc_score, accuracy_score
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F

import torch.optim as optim
from torch.optim import lr_scheduler
import torchvision
from torchvision import datasets, models, transforms
import random
from skimage import io, transform
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms
from torch.cuda.amp import autocast, GradScaler

from utils import *

# --- DATALOADERS ---
train_ds = datasets.ImageFolder(TRAIN_DIR, data_transforms['train'])
val_ds = datasets.ImageFolder(VAL_DIR, data_transforms['val'])

train_dl = DataLoader(train_ds, TRAIN_BATCH_SIZE, shuffle=True, num_workers=2)
val_dl = DataLoader(val_ds, VAL_BATCH_SIZE, shuffle=False, num_workers=2)
# -----------------------------------

# --- MODEL PREPARATION ---
def set_seed(seed = 42, set_torch=True):
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if set_torch:
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False 

class WoofNet(nn.Module):
    def __init__(self, backbone, n_classes, name='woofnet'):
        super(WoofNet, self).__init__()
        self.backbone = list(backbone.children())
        self.fc = nn.Linear(self.backbone[-1].in_features, n_classes)
        self.backbone = nn.Sequential(*self.backbone[:-1])
        self.name = name
    
    def forward(self, x):
        x = self.backbone(x).squeeze(-1).squeeze(-1)
        return self.fc(x)

backbone = models.resnet101(pretrained=True)
model = WoofNet(backbone, N_CLASSES, name='resnet101').to(device)

optimizer = optim.Adam(model.parameters(), LR)
loss_fn = nn.CrossEntropyLoss().to(device)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=EPOCHS * len(train_dl), eta_min=ETA_MIN)
# ----------------------

# Model fitting and evaluation function
import time
def train(model, train_dl, valid_dl, acc_fn, loss_fn=loss_fn, optimizer=optimizer, scheduler=scheduler, device=device, epochs=EPOCHS):
    set_seed(SEED)

    start = time.time()
    train_loss, valid_loss = [], []

    # MAIN LOOP
    for epoch in range(epochs):
        print('Epoch {}/{}'.format(epoch, epochs - 1))
        print('-' * 10)

        # ITERATE OVER TRAIN & EVAL MODES
        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train(True)
                dataloader = train_dl
            else:
                model.train(False)
                dataloader = valid_dl

            running_loss = 0.0
            running_acc = 0.0

            step = 0
            
            for x, y in dataloader:
                x = x.to(device)
                y = y.to(device)
                step += 1

                # forward pass
                if phase == 'train':
                    # zero the gradients
                    optimizer.zero_grad()

                    outputs = model(x)
                    loss = loss_fn(outputs, y)
                    
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                else:
                    with torch.no_grad():
                        outputs = model(x)
                        loss = loss_fn(outputs, y)

                acc = acc_fn(outputs.detach(), y)

                running_acc  += acc.cpu().item()*dataloader.batch_size
                running_loss += loss.detach().cpu().item()*dataloader.batch_size 

                # PRINT INFO EVERY 100 STEPS
                if step % 100 == 0:
                    print('Current step: {}  Loss: {}  Acc: {}  AllocMem (Mb): {}'.format(step, loss, acc, torch.cuda.memory_allocated()/1024/1024))
                    
            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_acc / len(dataloader.dataset)

            print('Epoch {}/{}'.format(epoch, epochs - 1))
            print('-' * 10)
            print('{} Loss: {:.4f} Acc: {}'.format(phase, epoch_loss, epoch_acc))
            print('-' * 10)
            
            # SAVE THE BEST MODEL
            if phase == 'valid' and valid_loss and epoch_loss < min(valid_loss):
                torch.save(model.state_dict(), 'model.bin')
            elif phase == 'valid' and not valid_loss:
                torch.save(model.state_dict(), 'model.bin')
            
            train_loss.append(epoch_loss) if phase=='train' else valid_loss.append(epoch_loss)

    time_elapsed = time.time() - start
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))    
    
    return train_loss, valid_loss

def accuracy_fn(logits, y_true):
    return torch.sum((torch.argmax(logits, 1) == y_true))/y_true.shape[0]

tloss, vloss = train(model, train_dl, val_dl, accuracy_fn)

# PLOT LOSS HISTORY
plt.figure(figsize=(10, 4))
plt.plot(tloss, label='Train loss')
plt.plot(vloss, label='Val loss')
plt.legend()
plt.show()