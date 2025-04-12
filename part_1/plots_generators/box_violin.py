import matplotlib.pyplot as plt
import seaborn as sns
from part_1.main_code import aux_funcs_and_vals as aux
from part_1.main_code.data_loader import loaded_fifa22_data as data

def box_plot(title, xlabel, ylabel, format, data, xdata, ydata, xfilter, xfilter_data):
    filtered_data = aux.filter_data_by_x(data, xfilter_data, xfilter)
    sns.boxplot(data=filtered_data, x=xdata, y=ydata)
    aux.format_plot(title, xlabel, ylabel, format)

def violin_plot(title, xlabel, ylabel, format, data, xdata, ydata, xfilter, xfilter_data, hue, hue_title, split):
    filtered_data = aux.filter_data_by_x(data, xfilter_data, xfilter)
    if hue is not None:
        ax = sns.violinplot(data=filtered_data, x=xdata, y=ydata, hue=hue, split=split, palette="Blues")
        ax.get_legend().set_title(hue_title)
    else:
        sns.violinplot(data=filtered_data, x=xdata, y=ydata, hue=hue, split=split)
    aux.format_plot(title, xlabel, ylabel, format)


if __name__ == "__main__":
    box_plot("Overall ratings in 'top5' leagues", "League", "Overall", aux.default_format,
             data, "league_name", "overall", aux.top_5_leagues, "league_name")
    plt.show()

    violin_plot("Overall vs preferred foot - wingers", "Position", "Overall", aux.default_format,
                 data, "club_position", "overall", aux.wingers, "club_position",
                "preferred_foot","Preferred foot", True)
    plt.show()