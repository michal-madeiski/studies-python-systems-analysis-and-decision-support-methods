import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt

def line_plot(path_to_csv, title, save_title):
    data = pd.read_csv(path_to_csv)
    data_melted = data.melt(id_vars="model", var_name="feature", value_name="score")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="model", y="score", data=data_melted, marker="o", hue="feature")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.8)
    plt.savefig(save_title + ".png")
    plt.close()

line_plot("num_comp.csv", "R2 score for selected features", "num_comp_plot")
line_plot("cat_comp.csv", "Accuracy score for selected features", "cat_comp_plot")