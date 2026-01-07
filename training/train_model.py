import importlib
import msc_dataset
import DS_CNN
import val
import train
from msc_dataset import MSCDataset
from DS_CNN import DSCNN
from train import train_and_validate

import numpy as np
import os
import pandas as pd
import random
import torch
import torchaudio.transforms as T
import itertools
from tqdm import tqdm

from time import time
from torch import nn

print(f'PyTorch version: {torch.__version__}')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'CUDA available: {torch.cuda.is_available()}')

torch.manual_seed(0)
np.random.seed(0)
random.seed(0)

CONFIG = {
    'frame_length_ms': 32,
    'overlap_percentage': 25,
    'n_mels': 10,
    'n_mfcc': 10,
    'f_min': 20,
    'f_max': 8000,
    'learning_rate': 0.001,
    'batch_size': 32,
    'train_steps': 2000
}

CLASSES = ["stop", "up"]


frame_length_in_s = CONFIG['frame_length_ms'] / 1000.0
frame_step_in_s = frame_length_in_s * (1 - CONFIG['overlap_percentage'] / 100.0)

transform = T.MFCC(
    sample_rate=16000,
    n_mfcc=CONFIG['n_mfcc'],
    log_mels=True,
    melkwargs=dict(
        n_fft=int(frame_length_in_s * 16000),
        win_length=int(frame_length_in_s * 16000),
        hop_length=int(frame_step_in_s * 16000),
        center=False,
        f_min=CONFIG['f_min'],
        f_max=CONFIG['f_max'],
        n_mels=CONFIG['n_mels'],
    )
)

train_ds = MSCDataset(root_dir='../msc_train', classes=CLASSES, transform=transform, augment=True)
eval_ds = MSCDataset(root_dir='../msc_val', classes=CLASSES, transform=transform, augment=False)

train_loader = torch.utils.data.DataLoader(train_ds, batch_size=CONFIG['batch_size'], shuffle=True, num_workers=0)
val_loader = torch.utils.data.DataLoader(eval_ds, batch_size=CONFIG['batch_size'], shuffle=False, num_workers=0)

model = DSCNN(num_classes=len(CLASSES), in_channels=1)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=CONFIG['learning_rate'])
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=800, gamma=0.1)

start_time = time()
best_weights, best_acc = train_and_validate(
    model, train_loader, val_loader, criterion, optimizer, scheduler, device,
    train_steps=CONFIG['train_steps']
)
print(f"\n Training completed in {time() - start_time:.0f}s. Best Val Accuracy: {best_acc:.2f}%")

model.load_state_dict(best_weights)
# torch.save(model.state_dict(), 'best_model.pth')
# print("model saved as 'best_model.pth'")

#export to ONNX
timestamp = "manu_model_96_10mels"

saved_model_dir = './'
if not os.path.exists(saved_model_dir):
    os.makedirs(saved_model_dir)

print(f'Model Timestamp: {timestamp}')
model_cpu = model.cpu()
torch.onnx.export(
    transform,  # model to export
    torch.randn(1, 1, 16000),  # inputs of the model,
    f'{saved_model_dir}/{timestamp}_frontend.onnx',  # filename of the ONNX model
    input_names=['input'],  # input name in the ONNX model
    dynamo=True,
    optimize=True,
    report=False,
    external_data=False,
)
torch.onnx.export(
    model_cpu,
    train_ds[0]['x'].unsqueeze(0).cpu(),
    f'{saved_model_dir}/{timestamp}_model.onnx',
    input_names=['input'],
    dynamo=False,
    opset_version=17,

)
