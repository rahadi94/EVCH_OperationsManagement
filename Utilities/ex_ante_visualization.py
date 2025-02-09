import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
from matplotlib import gridspec
import seaborn as sns
import Utilities.sim_output_processing as proc
import Utilities.sim_input_processing as prep
import warnings
from matplotlib.backends.backend_pdf import PdfPages

from Environment.helper.configuration.configuration import Configuration

warnings.filterwarnings("ignore", category=Warning)


def set_plotting_style(palette="mako"):
    # This sets reasonable defaults for font size for a figure that will go in a paper
    sns.set_context("paper")
    # Set the font to be serif, rather than sans
    sns.set(font="serif")

    # define colors
    sns.set_palette(palette, n_colors=9)
    # project_even_cols = [[44 / 255, 25 / 255, 149 / 255, .75], [147 / 255, 41 / 255, 223 / 255, .75],[0 / 255, 9 / 255, 43 / 255, .75]]

    # Make the background white, and specify the font family
    sns.set_style(
        "ticks", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]}
    )


title_list = [
    "Dynamic Capacity-based (Proposed Model)",
    "Traditional Dynamic",
    "Dynamic Menu-based",
    "Time-of-Use",
]


def plot_full_sim_horizon_combined_load_sink_curve(
    palette="mako", hue=None, visuals_save_path="visuals_output/", post_fix=""
):
    """
    Plots load curve in 1-timestep (i.e., 1 minute) resolution over entire sim horizon
    :param charging_load_data:
    :param palette:
    :param hue:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """

    set_plotting_style()

    # data_charging = charging_load_data
    # data_building = building_load_data
    # data_storage = storage_load_data
    # data_total = data_charging.merge(data_building, on="sim_time", how="left")
    # data_total = data_total.merge(data_storage, on="sim_time", how="left")
    # data_total["combined_total_load"] = data_total["total_load"]+data_total["load_kw_rescaled"]+data_total["charge_load"]
    (
        data_charging_list,
        data_building_list,
        data_storage_list,
        data_total_list,
        max_load_list,
        xmax_list,
    ) = ([], [], [], [], [], [])
    i = 0
    for agent_name in [
        "pricing_double_PV_p0_alpha_low",
        "pricing_double_PV_p0_low",
        "pricing_double_PV_discrete_low_new",
    ]:
        data_charging_list.append(
            pd.read_csv(visuals_save_path + agent_name + "_charging_data.csv")
        )
        data_building_list.append(
            pd.read_csv(visuals_save_path + agent_name + "_data_building.csv")
        )
        data_storage_list.append(
            pd.read_csv(visuals_save_path + agent_name + "_data_storage.csv")
        )
        data_total_list.append(
            pd.read_csv(visuals_save_path + agent_name + "_data_total.csv")
        )
        max_load_list.append(data_total_list[i]["combined_total_load"].max())
        xmax_list.append(data_total_list[i]["sim_time"].max())
        i += 1
    colors = sns.color_palette(palette, n_colors=8)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))
    for j in range(3):
        ax[j].plot(
            data_charging_list[j]["total_load"], color=colors[0], label="EV Load"
        )
        ax[j].plot(
            data_building_list[j]["load_kw_rescaled"],
            color=colors[2],
            label="Building Load",
        )
        ax[j].plot(
            data_storage_list[j]["charge_load"], color=colors[4], label="Storage Load"
        )
        ax[j].plot(
            data_total_list[j]["combined_total_load"], color=colors[6], label="Combined"
        )
        ax[j].set_xlabel("Simulation Time")
        ax[j].set_ylabel("Power [kW]")
        ax[j].set_ylim(0, 1000)
        ax[j].set_xlim((0 - 50, xmax_list[j] + 50))
        ax[j].set_title(title_list[j], fontsize=14)
        if j == 0:
            ax[j].legend(loc="upper right", ncol=2, fontsize=10)

        # define avg
        ax[j].hlines(
            y=max_load_list[j],
            xmin=0,
            xmax=xmax_list[j],
            colors="k",
            linestyles="dashed",
            label="Maximum Load = {}".format(round(max_load_list[j], 2)),
        )
        ax[j].text(
            0,
            max_load_list[j] - 100,
            "Max. Combined Load: \n {} kW".format(round(max_load_list[j], 2)),
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(visuals_save_path + f"joint_plots_loads.pdf")

    plt.show()


def plot_full_sim_horizon_combined_load_source_curve(
    palette="mako", hue=None, visuals_save_path="visuals_output/"
):
    """
    Plots load curve in 1-timestep (i.e., 1 minute) resolution over entire sim horizon
    :param charging_load_data:
    :param palette:
    :param hue:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """

    combined_df_list, max_grid_load_list, xmax_list = [], [], []
    for agent_name in [
        "pricing_double_PV_p0_alpha_low",
        "pricing_double_PV_p0_low",
        "pricing_double_PV_discrete_low_new",
    ]:
        combined_df = pd.read_csv(visuals_save_path + agent_name + "_supply.csv")
        if agent_name == "pricing_double":
            combined_df["pv_generation"] = 0
        combined_df_list.append(combined_df)
        max_grid_load_list.append(combined_df["grid_supply"].max())

    xmax = combined_df["sim_time"].max()

    set_plotting_style()

    colors = sns.color_palette(palette, n_colors=8)
    fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 6))

    for j in range(3):

        ax[j].plot(
            combined_df_list[j]["pv_generation"], color=colors[0], label="PV Generation"
        )
        ax[j].plot(
            combined_df_list[j]["discharge_load"],
            color=colors[2],
            label="Storage Discharge",
        )
        ax[j].plot(
            combined_df_list[j]["grid_supply"], color=colors[4], label="Grid Supply"
        )
        ax[j].plot(
            combined_df_list[j]["combined_total_load"],
            color=colors[6],
            label="Total Supply",
        )
        ax[j].set_xlabel("Simulation Time")
        ax[j].set_ylabel("Power [kW]")
        ax[j].set_xlim((0 - 50, xmax + 50))
        ax[j].set_title(title_list[j], fontsize=14)
        if j == 0:
            ax[j].legend(loc="upper right", ncol=2, fontsize=8)
        ax[j].set_ylim(0, 1000)

        # define avg
        ax[j].hlines(
            y=max_grid_load_list[j],
            xmin=0,
            xmax=xmax,
            colors="k",
            linestyles="dashed",
            label="Maximum Grid Load = {}".format(round(max_grid_load_list[j], 2)),
        )
        ax[j].text(
            0,
            max_grid_load_list[j] - 100,
            "Max. Grid Load: \n {} kW".format(round(max_grid_load_list[j], 2)),
            fontsize=10,
        )

    plt.tight_layout()

    plt.savefig(visuals_save_path + f"joint_plots_supply.pdf")

    plt.show()


def plot_full_sim_horizon_combined_load_source_curve_4(
    palette="mako", hue=None, visuals_save_path="visuals_output/"
):
    """
    Plots load curve in 1-timestep (i.e., 1 minute) resolution over entire sim horizon
    :param charging_load_data:
    :param palette:
    :param hue:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """

    combined_df_list, max_grid_load_list, xmax_list = [], [], []
    for agent_name in [
        "pricing_double_PV_p0_alpha_low",
        "pricing_double_PV_p0_low",
        "pricing_double_PV_discrete_low_new",
        "pricing_double_p0_alpha_ToU",
    ]:
        combined_df = pd.read_csv(visuals_save_path + agent_name + "_supply.csv")
        if agent_name == "pricing_double":
            combined_df["pv_generation"] = 0
        combined_df_list.append(combined_df)
        max_grid_load_list.append(combined_df["grid_supply"].max())

    xmax = combined_df["sim_time"].max()

    set_plotting_style()

    colors = sns.color_palette(palette, n_colors=8)
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 8))
    ax = ax.flatten()

    for j in range(4):

        ax[j].plot(
            combined_df_list[j]["pv_generation"], color=colors[0], label="PV Generation"
        )
        # ax[j].plot(combined_df_list[j]["discharge_load"], color=colors[2], label="Storage Discharge")
        ax[j].plot(
            combined_df_list[j]["grid_supply"], color=colors[4], label="Grid Supply"
        )
        ax[j].plot(
            combined_df_list[j]["combined_total_load"],
            color=colors[6],
            label="Total Supply",
        )
        ax[j].set_xlabel("Simulation Time")
        ax[j].set_ylabel("Power [kW]")
        ax[j].set_xlim((0 - 50, xmax + 50))
        # ax[j].set_title(title_list[j], fontsize=14, fontweight="bold")
        ax[j].set_title(title_list[j], fontsize=14)
        if j == 0:
            ax[j].legend(loc="upper right", ncol=2, fontsize=8)
        ax[j].set_ylim(0, 1200)

        # define avg
        # ax[j].hlines(y=max_grid_load_list[j], xmin=0, xmax=xmax, colors='k', linestyles='dashed',
        #              label='Maximum Grid Load = {}'.format(round(max_grid_load_list[j], 2)))
        ax[j].hlines(
            y=500,
            xmin=0,
            xmax=xmax,
            colors="k",
            linestyles="dashed",
            label="Maximum Grid Load = {}".format(round(max_grid_load_list[j], 2)),
        )
        # ax[j].text(0, max_grid_load_list[j] - 100,
        #            'Max. Grid Load: \n {} kW'.format(round(max_grid_load_list[j], 2)), fontsize=10)
        ax[j].text(
            0, 500 + 100, "Peak Threshold: \n {} kW".format(round(500, 0)), fontsize=10
        )

    plt.tight_layout()

    plt.savefig(visuals_save_path + f"joint_plots_supply.pdf")

    plt.show()


# plot_full_sim_horizon_combined_load_sink_curve()
# plot_full_sim_horizon_combined_load_source_curve()
plot_full_sim_horizon_combined_load_source_curve_4()
