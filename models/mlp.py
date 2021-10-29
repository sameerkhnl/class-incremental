import torch
from torch import nn
from continuum.metrics.metrics import accuracy
import numpy as np
from models.resnet import WideResNet_28_10_cifar

# Define model
class MLP(nn.Module):
    def __init__(self, out_dim=10, in_channel=1, img_sz=32, hidden_dim=256):
        def init_weights(m):
          if isinstance(m, nn.Linear):
            torch.nn.init.xavier_uniform_(m.weight)

        super(MLP, self).__init__()
        self.in_dim = in_channel * img_sz * img_sz
        self.linear_relu_stack = nn.Sequential(
            nn.Linear(self.in_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(inplace=True),
        )
        self.last = nn.Linear(hidden_dim, out_dim)

        self.linear_relu_stack.apply(init_weights)

        # torch.nn.init.xavier_uniform(self.linear_relu_stack[0].weight)
        # torch.nn.init.xavier_uniform(self.linear_relu_stack[2].weight)
        # torch.nn.init.xavier_uniform(self.linear_relu_stack[4].weight)

        # self.linear_relu_stack.apply(init_weights)
    def features(self,x):
        x = self.linear_relu_stack(x.view(-1, self.in_dim))
        return x

    def logits(self, x):
        x = self.last(x)
        return x

    def forward(self, x):
        x = self.features(x)
        x = self.logits(x)
        return x

def MLP100():
    return MLP(hidden_dim=100)

def MLP400():
    return MLP(hidden_dim=400)

def MLP800():
    return MLP(hidden_dim=800)

def train(dataloader, model, loss_fn, optimizer, device):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y,t) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y.long())
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # if batch % 100 == 0:
        #     loss, current = loss.item(), batch * len(X)
        #     print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


def eval(dataloader, model,device, logger, logger_subset='train'):
    model.eval()
    _logits = []
    # test_loss, correct = 0, 0
    with torch.no_grad():
        for i,(X, y,t) in enumerate(dataloader):
            X, y = X.to(device), y.to(device)
            logits = model(X)
            _logits.append(logits.cpu().numpy())
            predictions = logits.argmax(1)
            logger.add([predictions, y, t], subset = logger_subset)
            
    _logits = np.concatenate(_logits)
    return _logits
            
            
            # correct /= size

