import torch.nn as nn
import torch

class LinearRegression(nn.Module):
    
    def __init__(self, input_dim, output_dim, dataset):
        super(LinearRegression, self).__init__()
        self.dataset = dataset
        
        # define layers
        self.linear = nn.Linear(input_dim, output_dim, dtype=torch.double)
    

    def _eval_layers(self):
        self.linear.eval()
    
    def forward(self, x):
        return self.linear(x)


class LinearClassifier(nn.Module):
    def __init__(self, conf):
        super(LinearClassifier, self).__init__()
        self.num_classes = conf.num_classes
        self.linear = nn.Linear(conf.data_dim, conf.num_classes)
    
    def _eval_layers(self):
        self.linear.eval()
    
    def forward(self, x):
        return self.linear(x)


def linear_regression(conf):
    """Constructs a linear regression (y = Ax) model."""
    in_dim = conf.data_dim
    out_dim = 1
    return LinearRegression(in_dim, out_dim, dataset=conf.data)


def linear_classification(conf):
    """Constructs a linear model."""
    return LinearClassifier(conf)
