import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


class DatasetVisualization:
    def __init__(self):
        pass

    def get_descriptive_statistics(self, data):
        df = pd.DataFrame(data, columns=['Values'])
        print(df['Values'].describe())


    def dataset_distribution(self, dataset, font_size=10, save=False):

        documents_length_lst = [len(txt.split(" ")) for txt in dataset]

        self.get_descriptive_statistics(documents_length_lst)

        # Create a figure with GridSpec
        fig = plt.figure(figsize=(4, 4))  # Adjust the figure size as needed
        gs = gridspec.GridSpec(5, 1, hspace=0)
        # plt.suptitle('Distribution of The Token Counts', fontsize=12)
        plt.tight_layout(rect=[0, 0, 1, 0.93])  # rect parameter excludes suptitle from tight_layout adjustments

        # Plot histograms
        ax1 = fig.add_subplot(gs[0:4, 0])
        ax1.hist(documents_length_lst, bins=range(1, max(documents_length_lst) + 2), color = 'k', alpha=0.7)
        ax1.set_title('Distribution', fontsize=font_size)
        # ax1.set_xlabel('Number of Tokens', fontsize=font_size)
        ax1.set_ylabel('Frequency', fontsize=font_size)
        ax1.grid(axis='y', linestyle='--')
        ax1.tick_params(axis='both', which='major', labelsize=font_size)
        yticks = np.arange(0, max(np.histogram(documents_length_lst, bins=range(1, max(documents_length_lst) + 2))[0]), step=10)
        ax1.set_yticks(yticks)

        medianprops = dict(linestyle='-', linewidth=3, color='orange')  # Adjust 'linewidth' as needed

        # Plot box plots under each histogram
        ax4 = fig.add_subplot(gs[4, 0])
        ax4.boxplot(documents_length_lst, vert=False, medianprops=medianprops)
        ax4.set_xlabel('Number of Tokens', fontsize=font_size)
        ax4.set_yticks([])  # Remove y-ticks for a cleaner look
        ax4.tick_params(axis='both', which='major', labelsize=font_size)

        # ax1.set_xticks([])
        # ax1.set_xticklabels([])

        # Show the combined plot
        if save:
            plt.savefig("document_length_distribution.png", bbox_inches='tight', pad_inches=0, dpi=800)
        plt.show()
