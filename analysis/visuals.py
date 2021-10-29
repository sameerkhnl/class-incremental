import matplotlib.pyplot as plt
from matplotlib.pyplot import cm
import pandas as pd


def plot_line_chart(input_filepath,  title, output_filepath, colors=None):
    df = pd.read_csv(input_filepath, index_col=0)
    COLORS = ["grey", "indianred", "black", "deepskyblue", "blue", "yellowgreen", "goldenrod", "red", "darkblue", "brown", "crimson", "saddlebrown", "aqua", "olive", "mediumvioletred", "darkorange", "darkslategray"]
    if colors is None:
        colors = COLORS
    fig,ax = plt.subplots(figsize=(12,8))
    x = [i+1 for i in range(len(df.columns))]
    for (i,row),c in zip(df.iterrows(),colors):
        y = row.to_numpy()
        line = ax.plot(x,y, label=row.name, linewidth=2, color=c)
    leg = ax.legend(shadow=False, ncol=1)
    
    _ = ax.set_xticks(x)
    ax.set_xlabel('num tasks', fontsize=12)
    ax.set_ylabel('average accuracy (on tasks seen so far)', fontsize=12)
    ax.set_ylim((0,1))
    ax.set_title(title, fontsize=12)
    plt.xticks(rotation=75)
    plt.savefig(output_filepath, bbox_inches='tight')