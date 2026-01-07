import torch
from torch import nn

class DSCNN(nn.Module):
    def __init__(self, num_classes=2, in_channels=1):
        super(DSCNN, self).__init__()
        dim = 96
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, dim, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(dim),
            nn.ReLU(inplace=True),
        )
        self.ds_block1 = self._make_ds_block(dim, dim, stride=1)
        self.ds_block2 = self._make_ds_block(dim, dim, stride=1)
        self.ds_block3 = self._make_ds_block(dim, dim, stride=1)
        self.ds_block4 = self._make_ds_block(dim, dim, stride=1)

        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.3) 
        self.fc = nn.Linear(dim, num_classes)

    def _make_ds_block(self, in_ch, out_ch, stride=1):
        return nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, stride=stride, padding=1, groups=in_ch, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            
            nn.Conv2d(in_ch, out_ch, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.ds_block1(x)
        x = self.ds_block2(x)
        x = self.ds_block3(x)
        x = self.ds_block4(x)
        
        x = self.avg_pool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)
        return x