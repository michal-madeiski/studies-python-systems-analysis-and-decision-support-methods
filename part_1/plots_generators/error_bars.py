import matplotlib.pyplot as plt
import seaborn as sns
from part_1.main_code.data_loader import loaded_fifa22_data as data

def error_bar_plot(title, xlabel, data, feature, arg, **kws):
    f, axs = plt.subplots(2, sharex=True, layout="tight")
    sns.pointplot(x=data[feature], errorbar=arg, **kws, capsize=.3, ax=axs[0])
    plt.title(title)
    sns.stripplot(x=data[feature], jitter=.3, ax=axs[1])
    plt.xlabel(xlabel)


if __name__ == '__main__':
    error_bar_plot("Values distribution", "Value [eur]", data, "value_eur", "sd")
    plt.show()