import matplotlib.pyplot as plt

#FUNCS#
def format_plot(title, xlabel, ylabel, f):
    plt.title(title, fontsize=f["title_fontsize"], fontname=f["fontname"], fontweight=f["title_fontweight"], pad=f["pad"])
    plt.xlabel(xlabel, fontsize=f["fontsize"], fontname=f["fontname"], color=f["color"])
    plt.ylabel(ylabel, fontsize=f["fontsize"], fontname=f["fontname"], color=f["color"])
    plt.xticks(fontsize=f["fontsize"], fontname=f["fontname"], color=f["color"], rotation=f["xrotation"], ha=f["xha"])
    plt.yticks(fontsize=f["fontsize"], fontname=f["fontname"], color=f["color"])
    plt.tight_layout()

def save_plot(name):
    plt.savefig(f"../plots/{name}.png", dpi=300, bbox_inches="tight")
    plt.close()

def filter_data_by_x(data, xdata, xfilter):
    if xfilter is not None:
        return data[data[xdata].isin(xfilter)]
    else:
        return data
#FUNCS#

#VALS#
top_5_leagues = ["English Premier League", "Spain Primera Division", "French Ligue 1", "German 1. Bundesliga", "Italian Serie A"]
main_stats = ["pace", "shooting", "passing", "dribbling", "defending", "physic"]
wingers = ["LW", "LM", "RM", "RW"]
nations_hist = ["Poland", "France", "Brazil", "Belgium", "Sweden", "Egypt", "Costa Rica"]

default_format = {
    "fontname": "Arial",
    "color": "darkblue",
    "title_fontsize": 12,
    "fontsize": 8,
    "title_fontweight": "bold",
    "pad": 12,
    "xrotation": 45,
    "xha": "right"
}
#VALS#