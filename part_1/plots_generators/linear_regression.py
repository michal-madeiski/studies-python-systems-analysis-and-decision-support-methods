import matplotlib.pyplot as plt
import seaborn as sns
from part_1.main_code import aux_funcs_and_vals as aux
from part_1.main_code.data_loader import loaded_fifa22_data as data

def linear_regression_plot(title, xlabel, ylabel, format, data, xdata, ydata, xfilter, xfilter_data):
    filtered_data = aux.filter_data_by_x(data, xfilter_data, xfilter)
    sns.lmplot(data=filtered_data, x=xdata, y=ydata, scatter=True,  line_kws={"color": "orange"})
    aux.format_plot(title, xlabel, ylabel, format)


if __name__ == '__main__':
    linear_regression_plot("Pace vs height [cm]", "Pace", "Height [cm]", aux.default_format, data,
                           "pace", "height_cm", aux.top_5_leagues, "league_name")
    plt.show()