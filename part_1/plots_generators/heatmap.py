import matplotlib.pyplot as plt
import seaborn as sns
from part_1.main_code import aux_funcs_and_vals as aux
from part_1.main_code.data_loader import loaded_fifa22_data as data

def heatmap_plot(title, xlabel, ylabel, format, data, features):
    sns.heatmap(data[features].corr(), cmap = "crest", annot=True, fmt=".2f", linewidths=.5, vmin=-1, vmax=1)
    aux.format_plot(title, xlabel, ylabel, format)


if __name__ == '__main__':
    heatmap_plot("Base stats correlation", None, None, aux.default_format,
                 data, aux.main_stats)
    plt.show()