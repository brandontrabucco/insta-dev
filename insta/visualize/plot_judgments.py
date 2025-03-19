from insta.utils import (
    VALUE_KEYS
)

import argparse
import json
import os


def plot_judgments(judgments_dir, output_plot_name,output_dir):
    """Plot the statistics of judge predictions on data from Insta.

    Arguments:

    judgments_dir: str
        The directory containing the judgments.

    output_plot_name: str
        The name of the output plot.

    output_dir: str
        The directory to save the output plot.
    
    """

    import matplotlib.pyplot as plt
    import matplotlib
    from matplotlib import font_manager

    import seaborn as sns
    import pandas as pd

    font_files = font_manager.findSystemFonts(fontpaths=["msttcorefonts"])
    for font_file in font_files:
        font_manager.fontManager.addfont(font_file)

    matplotlib.rc('font', family = 'Times New Roman', serif = 'cm10')
    matplotlib.rc('mathtext', fontset = 'cm')

    all_judgment_files = os.listdir(
        judgments_dir
    )

    
    all_judgments = []

    for judgment_file in all_judgment_files:

        judgment_file = os.path.join(
            judgments_dir,
            judgment_file
        )

        with open(judgment_file, 'r') as file:
            judgment = json.load(file)

        all_judgments.append(judgment)

    records = []

    for judgment_dict in all_judgments:

        records.append({
            key: judgment_dict[key] or 0.0
            for key in VALUE_KEYS
        })

    data = pd.DataFrame.from_records(records)

    fig, axes = plt.subplots(
        1, len(VALUE_KEYS), figsize = (24, 4)
    )

    color_palette = sns.color_palette(
        "colorblind",
        len(VALUE_KEYS)
    )

    for i, metric_key in enumerate(VALUE_KEYS):

        axis = sns.histplot(
            x = metric_key,
            data = data,
            bins = 11,
            binwidth = 0.1,
            binrange = (-0.05, 1.05),
            stat = 'count',
            facecolor = color_palette[0],
            edgecolor = 'white',
            linewidth = 4,
            ax = axes[i]
        )

        handles, labels = axis.get_legend_handles_labels()
        axis.legend([],[], frameon = False)

        axis.set(xlabel = None)
        axis.set(ylabel = None)

        axis.set_ylim(
            0, 40_000
        )

        axis.spines['right'].set_visible(False)
        axis.spines['top'].set_visible(False)

        axis.xaxis.set_ticks_position('bottom')
        axis.yaxis.set_ticks_position('left')

        axis.yaxis.set_tick_params(
            labelsize = 16,
            pad = 10
        )
        axis.xaxis.set_tick_params(
            labelsize = 16,
            labelrotation = 45,
            pad = 10
        )

        if i == 0:

            axis.set_ylabel(
                "Task Count",
                fontsize = 24,
                fontweight = 'bold',
                labelpad = 12
            )

        axis.set_xlabel(
            metric_key.replace("_", " ").title(),
            fontsize = 24,
            fontweight = 'bold',
            labelpad = 12
        )

        axis.grid(
            color = 'grey',
            linestyle = 'dotted',
            linewidth = 2
        )

    plt.tight_layout(pad = 1.0)
    
    os.makedirs(
        output_dir,
        exist_ok = True
    )

    plt.savefig(
        os.path.join(
            output_dir,
            output_plot_name + ".png"
        ),
        dpi = 300,
        bbox_inches = 'tight',
        format = 'png'
    )

    plt.savefig(
        os.path.join(
            output_dir,
            output_plot_name + ".pdf"
        ),
        dpi = 300,
        bbox_inches = 'tight',
        format = 'pdf'
    )


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--judgments_dir",
        type = str,
        default = "data/judgments"
    )

    parser.add_argument(
        "--output_plot_name",
        type = str,
        default = "judge_statistics"
    )

    parser.add_argument(
        "--output_dir",
        type = str,
        default = "plots"
    )

    args = parser.parse_args()

    plot_judgments(
        judgments_dir = args.judgments_dir,
        output_plot_name = args.output_plot_name,
        output_dir = args.output_dir
    )
