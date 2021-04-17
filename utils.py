import torch
from torchvision import datasets, transforms

# --- CONSTANTS ---
SEED = 1337
TRAIN_BATCH_SIZE = 32
VAL_BATCH_SIZE = 256

TRAIN_DIR = 'imagewoof2/train/'
VAL_DIR  = 'imagewoof2/val/'

N_CLASSES = len(os.listdir(TRAIN_DIR))
EPOCHS = 30
ETA_MIN = 3e-6
LR = 3e-4

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# -----------------

# Transforms
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomHorizontalFlip(p=.5),
        transforms.RandomVerticalFlip(p=.5),
        transforms.RandomResizedCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Eval dataset
class TestDataset(Dataset):
    def __init__(self, filenames, transforms):
        self.transform = transforms
        self.filenames = filenames

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, index):        
        image = Image.open(self.filenames[index]).convert('RGB')
        image = self.transform(image)
        return torch.tensor(image, dtype=torch.float32)
