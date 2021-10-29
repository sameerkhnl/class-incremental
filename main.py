import argparse
import copy
import csv
import os

import os, ssl
from pathlib import Path

from analysis.outputs import append_metrics_to_file, initialise_file, print_summary_to_file
from analysis.visuals import plot_line_chart

if (not os.environ.get('PYTHONHTTPSVERIFY', '') and
getattr(ssl, '_create_unverified_context', None)):
    ssl._create_default_https_context = ssl._create_unverified_context

import matplotlib.pyplot as plt
import numpy as np
import torch
from continuum import (Logger,
                       rehearsal)
from continuum.tasks import split_train_val
from torch import nn
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

import utils
from dataloaders.base import (CIFAR10_scenario, CIFAR100_scenario,
                              MNIST_scenario)
from models.mlp import MLP400, eval, train
from models.resnet import WideResNet_28_2_cifar
import sys
# shutil.rmtree('./data')

class CustomHerding:
    # z is the indices
  def __call__(self, x: np.ndarray, y: np.ndarray, t: np.ndarray, dict_vals, nb_per_class) -> np.ndarray:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    y_hat = dict_vals['logits']
    x,y,t,y_hat = torch.from_numpy(x).to(device),torch.from_numpy(y).to(device),torch.from_numpy(t).to(device),torch.from_numpy(y_hat).to(device)
    idx_correct = torch.nonzero(y_hat.argmax(1) == y)
    idx_correct = torch.flatten(idx_correct)
    x, y, t, y_hat = x[idx_correct], y[idx_correct], t[idx_correct], y_hat[idx_correct]
    top_two = torch.topk(y_hat, 2, dim=1)[0]
    abs = torch.abs(torch.diff(top_two))
    flattened = torch.flatten(abs)
    indices_sorted = torch.sort(flattened)[1]
    x, y, t = x[indices_sorted],y[indices_sorted],t[indices_sorted]

    reverse = True if dict_vals['herding_method'] == 'cl_margin_reverse' else False
    indexes = torch.empty(0, dtype=torch.long).to(device)
    for class_id in torch.unique(y):
        class_indices = torch.where(y == class_id)[0]
        if reverse:
            class_indices = class_indices[-nb_per_class:]
        else:
            class_indices = class_indices[:nb_per_class]
        indexes = torch.cat((indexes,class_indices))
    x,y,t = x[indexes].cpu().numpy(), y[indexes].cpu().numpy(), t[indexes].cpu().numpy()
    return x,y,t

def run(args):
    path = args.data_path
    scenarios = args.scenarios
    N_TASKS = args.tasks
    N_CLASSES = args.classes
    model = args.model
    device = args.device
    MEMORY_SIZE = args.memory
    batch_size = args.batch_size
    model = model(out_dim=N_CLASSES).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    iters = args.iters
    n_workers = args.workers
    loss_fn = args.loss_fn()
    debug = args.debug
    dataset = args.dataset
    replay = args.replay
    train_scenario, test_scenario = scenarios
    agent_name = None
    memory = None
    seed = args.seed
    filepath = args.filepath_main

    if replay:
        if not hasattr(args,'herding_method') or args.herding_method not in ['random', 'cl_margin', 'barycenter', 'cluster', 'cl_margin_reverse']:
            raise Exception(f'Please define one of the following herding methods when using memory: random, cl_margin, cl_margin_reverse, barycenter, cluster')
        herding_method = args.herding_method
        agent_name = herding_method

        nb_per_class = int(MEMORY_SIZE/N_CLASSES)
        if herding_method == 'random':
            memory = rehearsal.RehearsalMemory(memory_size=MEMORY_SIZE, herding_method='random', fixed_memory=True, nb_total_classes=N_CLASSES)
        elif herding_method == 'cl_margin' or herding_method == 'cl_margin_reverse':
            herding_cl_margin = CustomHerding()
            memory = rehearsal.RehearsalMemory(memory_size=MEMORY_SIZE, herding_method=herding_cl_margin, fixed_memory=True, nb_total_classes=N_CLASSES) 
        elif herding_method == 'barycenter':
            memory = rehearsal.RehearsalMemory(memory_size=MEMORY_SIZE, herding_method='barycenter', fixed_memory=True, nb_total_classes=N_CLASSES)
        elif herding_method == 'cluster':
            memory = rehearsal.RehearsalMemory(memory_size=MEMORY_SIZE, herding_method='cluster', fixed_memory=True, nb_total_classes=N_CLASSES)

        print(f'Using rehearsal memory of size {MEMORY_SIZE}, #examples per class: {nb_per_class}, #classes: {N_CLASSES}')
        print(f'herding method: {herding_method}')
    else:
        if  args.offline_training:
            print(f'Training offline (upperbound)...')
            agent_name = 'offline'
            # filename = filename + f'offline.csv'
        else:
            print(f'Training online without rehearsal (lowerbound)...')
            agent_name = 'baseline'
            # filename = filename + f'baseline.csv'
    print('')

    overall_acc_list = []

    for tid,taskset in enumerate(train_scenario):
        logger = Logger(list_subsets=['train','test'])

        train_dataloader = DataLoader(taskset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=n_workers)
        test_taskset = test_scenario[:tid + 1]
        test_dataloader = DataLoader(test_taskset, batch_size = len(test_taskset), num_workers=n_workers)

        #collect original values before exemplars are added
        taskset_original = copy.deepcopy(taskset)

        if memory is not None and len(memory) > 0:
                mem_x, mem_y, mem_t = memory.get()
                taskset.add_samples(mem_x,mem_y,mem_t)
        model.train()
        #train for iterations 
        for i in range(0,iters, len(train_dataloader)):
            if (i + len(train_dataloader)) > iters:
                remaining = iters - i
                subset_indices = np.random.randint(0, len(taskset), remaining * batch_size )
                new_taskset = Subset(taskset, subset_indices)
                train_dataloader = DataLoader(new_taskset, batch_size=batch_size, shuffle=True, drop_last=True)
            train(train_dataloader, model, loss_fn, optimizer, device) 
        
        train_acc_dataloader = DataLoader(taskset_original, batch_size = len(taskset), shuffle=False, num_workers=n_workers)
        logits = eval(train_acc_dataloader, model, device, logger, logger_subset = 'train') 
        print(f'Task: {tid}, Online accuracy: {logger.online_accuracy:.3f}')
        if memory is not None:
            # x,y,t = get_exemplars(taskset=taskset, herding_method=herding_method,predictions_all= dict_results['predictions_all'], correct_ex_indices=dict_results['correct_indices'], nb_per_class=nb_per_class, device=device)

            if herding_method == 'cl_margin' or herding_method == 'cl_margin_reverse':
                dict_vals = {'logits': logits, 'herding_method': herding_method}
                x,y,t = taskset_original.get_raw_samples()
                memory.add(x,y,t,dict_vals)
            else:
                features = extract_features(taskset_original, model, device)
                x,y,t = taskset_original.get_raw_samples()
                memory.add(x,y,t,features)

        eval(test_dataloader, model, device, logger, logger_subset='test')
        print('Testing on previous tasks:...')

        for i in range(tid+1):
            print(f"    Task {i} acc: {(logger.accuracy_per_task[i]):.3f}")
        avg_acc_per_task = np.mean(logger.accuracy_per_task)
        overall_acc_list.append(avg_acc_per_task)
        print(f'    Average accuracy: {avg_acc_per_task:.3f}')
        # print(f'    Average accuracy2: {logger.accuracy:.3f}')
        print('') 
        logger.end_task()
    
    overall_acc = np.array(overall_acc_list)
    overall_avg_acc = np.mean(overall_acc)
    print(f'Overall average accuracy: {overall_avg_acc:.3f}')
    overall_acc_list = [round(i,3) for i in overall_acc_list]
    append_metrics_to_file(filepath, overall_acc_list, agent_name, seed)
    return model

def extract_features(taskset, model, device):
    model = model.to(device)
    model.eval()
    dataloader = DataLoader(taskset, batch_size = len(taskset), shuffle=False)
    features = []
    with torch.no_grad():
        for x,y,t in dataloader:
            x,y = x.to(device), y.to(device)
            outputs = model.extract_features(x)
            features.append(outputs.cpu().numpy())
    
    features = np.concatenate(features)
    return features


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using {} device".format(device))

    data_path = Path.cwd() / 'data'
    
    args.data_path = data_path
    #currently des not work for MNIST
    cfg = {'MNIST': {
                'name': 'MNIST',
                'scenarios': MNIST_scenario,
                'model': MLP400,
                'n_classes': 10
            }, 
            'CIFAR10': {
                'name': 'CIFAR10',
                'scenarios': CIFAR10_scenario,  
                'model':WideResNet_28_2_cifar,               
                'n_classes': 10
            },
            'CIFAR100': {
                'name': 'CIFAR100',
                # 'scenarios': CIFAR100_scenario(path, 20, debug=debug), 
                'scenarios': CIFAR100_scenario, 
                'model': WideResNet_28_2_cifar,
                'n_classes': 100
            }}

    loss_fn = nn.CrossEntropyLoss
    # epochs= 1 if debug else 10

    args.loss_fn = loss_fn
    args.model = cfg[args.dataset]['model']
    args.device = device
    args.scenarios = cfg[args.dataset]['scenarios'](args.data_path, args.tasks, args.debug)
    args.iters = 50 if args.debug else 2000
    args.model = cfg[args.dataset]['model']
    args.classes = cfg[args.dataset]['n_classes']
    args.exp_info = f'{args.dataset}_t{args.tasks}_i{args.iters}_m{args.memory}_lr{args.lr}'

    results_path = Path.cwd() / 'results'
    store_path = Path.cwd() / 'store'

    if not results_path.exists():
        os.mkdir(results_path)
    if not store_path.exists():
        os.mkdir(store_path)
    
    models_path = Path.cwd() / 'store' / 'models'

    if not models_path.exists():
        os.mkdir(models_path)
    args.store_path = store_path
    args.models_path = models_path
    results_dataset_path = Path.cwd() / 'results' / args.dataset

    if not results_dataset_path.exists():
        os.mkdir(results_dataset_path)

    args.results_dataset_path = results_dataset_path

    filename_main = f'{args.exp_info}_all.csv'

    filepath_main = results_dataset_path / filename_main
    initialise_file(filepath_main, args.tasks)
    args.filepath_main = filepath_main
    filename_summary = f'{args.exp_info}_summary.csv'

    filepath_summary = results_dataset_path / filename_summary
    # args = {'loss_fn': loss_fn, 'model': cfg[dataset], 'device': device, 'batch_size': batch_size, 'debug': debug, 'config_dataset': cfg[dataset], 'iters': 50 if debug else 2000}

    run_all(args)    
    print_summary_to_file(filepath_main, filepath_summary)
    filename_lineplot = f'{args.dataset}_t{args.tasks}_i{args.iters}_m{args.memory}_line.pdf'
    filepath_lineplot = results_dataset_path / filename_lineplot

    line_title = f'{args.dataset} Class Incremental - {args.tasks} tasks'

    plot_line_chart(input_filepath=filepath_summary, title=line_title, output_filepath=filepath_lineplot)
        

def run_all(args):

    for i in range(args.repeat):
        np.random.seed(i)
        torch.manual_seed(i)
        args.offline = False
        args.replay = False
        args.seed = i
        #baseline
        model = run(args)
        model_path = args.models_path / f'{args.exp_info}_baseline.pt'
        torch.save(model, model_path)

        args.replay = True
        args.herding_method = 'cl_margin'
        model = run(args)
        model_path = args.models_path / f'{args.exp_info}_cl_margin.pt'
        torch.save(model, model_path)

        args.herding_method = 'cl_margin_reverse'
        model = run(args)
        model_path = args.models_path / f'{args.exp_info}_cl_margin_reverse.pt'
        torch.save(model, model_path)

    #using memory, herding type: random
        args.herding_method = 'random'
        model = run(args)
        model_path = args.models_path / f'{args.exp_info}_random.pt'

    # #using memory, herding type: barycenter
        args.herding_method = 'barycenter'
        model = run(args)
        model_path = args.models_path / f'{args.exp_info}_barycenter.pt'

    # #using memory, herding type: cluster
        args.herding_method = 'cluster'
        model = run(args)
        model_path = args.models_path / f'{args.exp_info}_barycenter.pt'

def get_args(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='CIFAR10(default) | CIFAR100')
    parser.add_argument('--tasks', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--momentum', type=float, default=0)
    parser.add_argument('--weight_decay', type=float, default=0)
    parser.add_argument('--schedule', nargs='+', type=int, default=[2], help='The list of epoch numbers to reduce learning rate by factor of 0.1. Last number is the end epoch')
    parser.add_argument('--print_freq', type=float, default=100, help="Print the log at every x iteration")
    parser.add_argument('--offline_training', dest='offline_training', default=False, action='store_true',
                        help="Non-incremental learning by make all data available in one batch. For measuring the upperbound performance.")
    parser.add_argument('--workers', type=int, default=3, help='# of threads for dataloader')
    parser.add_argument('--debug', default=False, action='store_true')
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--repeat', type=int, default=1, help='# of experiments')
    parser.add_argument('--memory', type=int, default=200, help='memory size for replay')
    args = parser.parse_args(argv)
    return args

if __name__ == '__main__':
    # torch.manual_seed(0)
    # np.random.seed(0)
    args = get_args(sys.argv[1:])
    main(args)


    