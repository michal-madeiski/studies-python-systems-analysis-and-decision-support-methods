import matplotlib.pyplot as plt
import seaborn as sns
from part_1.main_code import aux_funcs_and_vals as aux
from part_1.main_code.data_loader import loaded_fifa22_data as data

def bar_plot(title, xlabel, ylabel, format, data, xdata, ydata, xfilter, xfilter_data, type):
    filtered_data = aux.filter_data_by_x(data, xfilter_data, xfilter)
    if type is not None:
        match type:
            case "mean": filtered_data = filtered_data.groupby(xdata)[ydata].mean().reset_index()
            case "sum": filtered_data = filtered_data.groupby(xdata)[ydata].sum().reset_index()
            case "median": filtered_data = filtered_data.groupby(xdata)[ydata].median().reset_index()
            case "std": filtered_data = filtered_data.groupby(xdata)[ydata].std().reset_index()
            case _: filtered_data = filtered_data.groupby(xdata)[ydata].count().reset_index()
    sns.barplot(filtered_data, x=xdata, y=ydata)
    aux.format_plot(title, xlabel, ylabel, format)

def hist_plot(title, xlabel, ylabel, format, data, xdata, ydata, xfilter, xfilter_data, discrete, hue, hue_title):
    filtered_data = aux.filter_data_by_x(data, xfilter_data, xfilter)
    if hue is not None:
        ax = sns.histplot(data=filtered_data, x=xdata, y=ydata, discrete=discrete, hue=hue, palette="Blues_d")
        ax.get_legend().set_title(hue_title)
    else:
        sns.histplot(data=filtered_data, x=xdata, y=ydata, discrete=discrete, hue=hue)
    aux.format_plot(title, xlabel, ylabel, format)


if __name__ == '__main__':
    hist_plot("Distribution of jersey numbers in clubs", "Jersey number", "Count",
              aux.default_format, data, "club_jersey_number", None,None, None,
              True, None, None)
    plt.show()

    bar_plot("Average overall in certain countries", "Country", "Average overall",
             aux.default_format, data, "nationality_name", "overall", aux.nations_hist,
             "nationality_name", "mean")
    plt.show()

    hist_plot("Number of players in certain countries", "Country", "Count", aux.default_format,
              data,"nationality_name", None, aux.nations_hist, "nationality_name",
              True, None,None)
    plt.show()

    hist_plot("Dribbling vs skill moves", "Dribbling", "Overall", aux.default_format, data,
              "dribbling", None,None, None, True, "skill_moves",
              "Skill moves")
    plt.show()