import aux_funcs_and_vals as aux
from data_loader import loaded_fifa22_data as data
from part_1.plots_generators.box_violin import box_plot, violin_plot
from part_1.plots_generators.error_bars import error_bar_plot
from part_1.plots_generators.hist import bar_plot, hist_plot
from part_1.plots_generators.heatmap import heatmap_plot
from part_1.plots_generators.linear_regression import linear_regression_plot


if __name__ == "__main__":
    box_plot("Overall ratings in 'top5' leagues", "League", "Overall", aux.default_format,
             data, "league_name", "overall", aux.top_5_leagues, "league_name")
    aux.save_plot("box_overall_in_top5")

    violin_plot("Overall vs preferred foot - wingers", "Position", "Overall", aux.default_format,
                 data, "club_position", "overall", aux.wingers, "club_position",
                "preferred_foot","Preferred foot", True)
    aux.save_plot("violin_wingers_preferred_foot")

    error_bar_plot("Values distribution", "Value [eur]", data,
                   "value_eur", "sd")
    aux.save_plot("errorbar_value")

    hist_plot("Distribution of jersey numbers in clubs", "Jersey number", "Count",
              aux.default_format, data, "club_jersey_number", None,None, None,
              True, None, None)
    aux.save_plot("hist_jersey_number")

    bar_plot("Average overall in certain countries", "Country", "Average overall",
             aux.default_format, data, "nationality_name", "overall", aux.nations_hist,
             "nationality_name", "mean")
    aux.save_plot("bar_avg_overall_in_countries")

    hist_plot("Number of players in certain countries", "Country", "Count", aux.default_format,
              data,"nationality_name", None, aux.nations_hist, "nationality_name",
              True, None,None)
    aux.save_plot("hist_num_of_players_in_countries")

    hist_plot("Dribbling vs skill moves", "Dribbling", "Count", aux.default_format, data,
              "dribbling", None,None, None, True, "skill_moves",
              "Skill moves")
    aux.save_plot("hist_dribbling_vs_skill_moves")

    heatmap_plot("Base stats corelation", None, None, aux.default_format,
                 data, aux.main_stats)
    aux.save_plot("heatmap_base_stats")

    linear_regression_plot("Pace vs height [cm]", "Pace", "Height [cm]", aux.default_format, data,
                           "pace", "height_cm", aux.top_5_leagues, "league_name")
    aux.save_plot("lin_reg_pace_vs_height")