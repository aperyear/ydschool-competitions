import timm
import torch.nn as nn
import torch.nn.init as init


class MyModel(nn.Module):
    def __init__(self, n_classes, model_name):
        super(MyModel, self).__init__()
        self.feature = timm.create_model(model_name, pretrained=False)
        
        if model_name == 'inception_v4':
            self.out_features = self.feature.last_linear.in_features
            self.feature.last_linear = nn.Linear(in_features=self.out_features, out_features=n_classes, bias=True) 
            
        elif 'darknet53' in model_name:
            self.out_features = self.feature.head.fc.in_features
            self.feature.head.fc = nn.Linear(in_features=self.out_features, out_features=n_classes, bias=True) 
            
        elif 'dense' in model_name:
            self.out_features = self.feature.classifier.in_features
            self.feature.classifier = nn.Linear(in_features=self.out_features, out_features=n_classes, bias=True) 
            
        else:
            self.out_features = self.feature.fc.in_features
            self.feature.fc = nn.Linear(in_features=self.out_features, out_features=n_classes, bias=True) 
    
    def forward(self, x):
        x = self.feature(x)
        return x


def init_weight(model, kind='xavier'):
    for name, i in model.named_parameters():
        if kind == 'xavier':
            if i.dim() < 2:
                continue
            if 'weight' in name:
                init.xavier_normal_(i, gain=1.0)
            elif 'bias' in name:
                init.xavier_uniform_(i, gain=1.0)
            else:
                pass
        elif kind == 'kaiming':
            if i.dim() < 2:
                continue
            if 'weight' in name:
                init.kaiming_normal_(i)
            elif 'bias' in name:
                init.kaiming_uniform_(i)
            else:
                pass