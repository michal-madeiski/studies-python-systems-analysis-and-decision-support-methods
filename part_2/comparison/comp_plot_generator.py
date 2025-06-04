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
    plt.savefig("plots/" + save_title + ".png")
    plt.close()

def line_plot_mse(path_to_csv, title, save_title):
    data = pd.read_csv(path_to_csv)
    data = data.drop(index=1) #drop value_eur to for better plot visualisation
    data_melted = data.melt(id_vars="target", var_name="epoch", value_name="mse")
    plt.figure(figsize=(10, 6))
    sns.lineplot(x="epoch", y="mse", data=data_melted, marker="o", hue="target")
    plt.title(title)
    plt.xticks(rotation=45)
    plt.xticks([])
    plt.tight_layout()
    plt.legend(loc="upper left", bbox_to_anchor=(1, 1))
    plt.subplots_adjust(right=0.8)
    plt.savefig("plots/" + save_title + ".png")
    plt.close()

line_plot("csv_results/num_comp.csv", "R2 score for selected features", "num_comp_plot")
line_plot("csv_results/cat_comp.csv", "Accuracy score for selected features", "cat_comp_plot")
line_plot_mse("csv_results/all_mse.csv", "MSE by epoch", "all_mse_plot")
#line_plot_mse("all_mse_poly.csv", "MSE by epoch", "all_mse_poly_plot")