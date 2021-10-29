
from continuum.datasets.pytorch import CIFAR10, CIFAR100
import torch
from continuum import ClassIncremental
from continuum.tasks import split_train_val, TaskSet
from continuum.datasets import MNIST
from torch.utils.data import DataLoader, Subset
import torchvision
from torchvision import transforms
import numpy as np


def MNIST_scenario(path, n_tasks, debug=False):
  train_d, test_d = get_train_test(path, 'MNIST', debug)
  train_scenario, test_scenario = get_scenario(train_d, test_d, 'MNIST',n_tasks)
  return train_scenario, test_scenario

def CIFAR10_scenario(path, n_tasks, debug=False):
  train_d, test_d = get_train_test(path, 'CIFAR10', debug)
  train_scenario, test_scenario = get_scenario(train_d, test_d, 'CIFAR10',n_tasks)
  return train_scenario, test_scenario

def CIFAR100_scenario(path, n_tasks, debug=False):
  train_d, test_d = get_train_test(path, 'CIFAR100', debug)
  train_scenario, test_scenario = get_scenario(train_d, test_d, 'CIFAR100',n_tasks)
  return train_scenario, test_scenario

def get_train_test(path, dataset:str, debug=False):
  train_dataset = test_dataset = None

  if dataset == 'MNIST' :
    train_dataset = MNIST(path, download=True, train=False)
    test_dataset = MNIST(path, download=True, train=False)
  elif dataset == 'CIFAR10':
    train_dataset = CIFAR10(path, download=True, train=True)
    test_dataset = CIFAR10(path, download=True, train=False)
  elif dataset == 'CIFAR100':
    train_dataset = CIFAR100(path, download=True, train=True)
    test_dataset = CIFAR100(path, download=True, train=False)

  #when in debug mode, only take 2000 random samples
  if debug:
    idxs = np.random.randint(0,len(train_dataset.dataset), size=10000)
    train_dataset.dataset.data = train_dataset.dataset.data[idxs]
    train_dataset.dataset.targets = np.array(train_dataset.dataset.targets)
    train_dataset.dataset.targets = train_dataset.dataset.targets[idxs]

    idxs = np.random.randint(0, len(test_dataset.dataset), size=5000)
    test_dataset.dataset.data = test_dataset.dataset.data[idxs]
    test_dataset.dataset.targets = np.array(test_dataset.dataset.targets)
    test_dataset.dataset.targets = test_dataset.dataset.targets[idxs]
  
  return train_dataset, test_dataset
  
def get_scenario(train_dataset, test_dataset, dataset:str, n_tasks = 5):
    train_transform = test_transform = None

    if dataset == 'MNIST':
      normalize = transforms.Normalize(mean=(0.1000,), std=(0.2752,)) 
      #padding applied on MNIST, similar to the implementation by https://github.com/GT-RIPL/#Continual-Learning-Benchmark
      train_transform = transforms.Compose([
            transforms.Pad(2, fill=0, padding_mode='constant'),
            transforms.ToTensor(),
            normalize,
        ])
    elif dataset == 'CIFAR10' or dataset == 'CIFAR100':
       train_transform = transforms.Compose([
            transforms.ToTensor(),
        ])
    test_transform = train_transform
    scenario_train = ClassIncremental(train_dataset, nb_tasks=n_tasks, transformations=[train_transform])
    scenario_test = ClassIncremental(test_dataset, nb_tasks=n_tasks, transformations=[test_transform])

    return scenario_train, scenario_test 




