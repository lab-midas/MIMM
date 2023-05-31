import torch.nn as nn
import torch.nn.functional as F
import config.Load_Parameter


class FeatureEncoderNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_features = config.Load_Parameter.params.feature_vector_length
        self.classes_PT = 2
        self.classes_SC = 2
        self.classes = self.classes_PT+self.classes_SC
        self.fc = nn.Linear(256, 256)
        self.fc_pt_sc = nn.Linear(256, self.classes)

        self.features = nn.Sequential(
            nn.Conv2d(1, 6, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(6, 16, 3, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 3, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )

    def forward(self, X):
        X = self.features(X)
        X = X.view(-1, X.shape[1]*X.shape[2]*X.shape[3])
        X = F.relu(self.fc(X))

        if self.in_features == 4:
            X = self.fc_pt_sc(X)
        return X


class FeatureSelectionNetworkExtended(nn.Module):
    def __init__(self):
        super().__init__()
        self.classes_CL = config.Load_Parameter.params.numberClasses
        self.classes_CF = config.Load_Parameter.params.numberClassesConfounding
        self.classes = config.Load_Parameter.params.feature_vector_length

        
        self.in_channel = 1
        self.out_channel = 64*6*6

        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channel,
                      out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.fc1 = nn.Linear(in_features=self.out_channel, out_features=600)
        self.drop = nn.Dropout2d(0.25)
        self.fc2 = nn.Linear(in_features=600, out_features=120)
        self.fc3 = nn.Linear(in_features=120, out_features=self.classes)

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = out.view(out.size(0), -1)
        out = self.fc1(out)
        out = self.drop(out)
        out = self.fc2(out)
        out = self.fc3(out)
        return out
