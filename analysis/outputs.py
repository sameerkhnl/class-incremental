import os
import numpy as np
import pandas as pd
from pathlib import Path
import sys
import csv

def initialise_file(filepath, n_tasks):
    header = [f'task{i+1}' for i in range(n_tasks)]
    header.append('agent')
    header.append('seed')
    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)

def append_metrics_to_file(filepath, overall_acc_list, agent_name, seed):
    overall_acc_list.append(agent_name)
    overall_acc_list.append(seed)
    with open(filepath, 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(overall_acc_list)

def print_summary_to_file(raw_filepath, summary_filepath):
    df = pd.read_csv(raw_filepath).round(3)
    df = df.iloc[:,:-1]
    summary = df.groupby(['agent']).mean().round(3)
    summary.to_csv(summary_filepath)