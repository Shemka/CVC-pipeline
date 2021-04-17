import pandas as pd
import numpy as np
import cv2
import os
import glob
from PIL import Image
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

class Pipeline:
    def __init__(self, model, batch_size, device):
        self.batch_size = batch_size
        self.model = model.to(device).eval()

    def _prepare_dataset(self, filenames):
        self.ds = TestDataset(filenames, data_transforms['val'])
        self.dl = DataLoader(self.ds, self.batch_size, shuffle=False)

    def classify_image(self, filename):
        image = Image.open(filename).convert('RGB')
        image = data_transforms['val'](image).unsqueeze(0)
        image = image.to(device)
        with torch.no_grad():
            return torch.argmax(self.model(image), 1).cpu().item()
        
    def classify_images(self, filenames):
        self._prepare_dataset(filenames)
        predictions = None
        with torch.no_grad():
            for x in self.dl:
                x = x.to(device)
                logits = model(x)
                p = torch.argmax(logits, 1).cpu().view(-1).numpy()

                if predictions is None:
                    predictions = p
                else:
                    predictions = np.concatenate((predictions, p), axis=0)
            return predictions


if __name__ == '__main__':
    filenames = glob.glob('imagewoof2/val/*/*.JPEG')

    backbone = models.resnet101(pretrained=False)
    model = WoofNet(backbone, N_CLASSES, name='resnet101').to(device)
    model.load_state_dict(torch.load('model.bin'))

    pipeline = Pipeline(model, VAL_BATCH_SIZE, device)
    preds = pipeline.classify_images(filenames)

    print(preds)