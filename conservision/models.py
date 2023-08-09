from torch import nn
import torchvision.models as models

def resnet50_animal():
    model = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential( # type: ignore
        nn.Linear(2048, 100),  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(100, 8),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    return model


def resent152_animal():
    model = models.resnet152(weights=models.ResNet50_Weights.DEFAULT)
    model.fc = nn.Sequential( # type: ignore
        nn.Linear(2048, 100),  # dense layer takes a 2048-dim input and outputs 100-dim
        nn.ReLU(inplace=True),  # ReLU activation introduces non-linearity
        nn.Dropout(0.1),  # common technique to mitigate overfitting
        nn.Linear(100, 8),  # final dense layer outputs 8-dim corresponding to our target classes
    )
    return model