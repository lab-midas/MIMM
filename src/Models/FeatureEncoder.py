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
        self.fc_pt_sc = nn.Linear(256,self.classes)

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