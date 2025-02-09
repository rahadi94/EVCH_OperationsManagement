# Plotting Routines
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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
    sns.set_palette(palette, n_colors=4)
    # project_even_cols = [[44 / 255, 25 / 255, 149 / 255, .75], [147 / 255, 41 / 255, 223 / 255, .75],[0 / 255, 9 / 255, 43 / 255, .75]]

    # Make the background white, and specify the font family
    sns.set_style(
        "ticks", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]}
    )


def get_visuals(
    model,
    palette="mako",
    full_version=False,
    raw_output_save_path="./Utilities/raw_output/",
    visuals_save_path="./Utilities/visuals_output/",
    post_fix="",
    sim_start_date="2019-06-03",
):
    """
    Combined full visualization routine
    :param palette:
    :return
    """
    set_plotting_style()

    # process required results data
    df_building_load = model.base_load
    df_PV_generation = model.non_dispatchable_generator.generation_profile_actual
    # df_requests = proc.get_requests(raw_output_save_path=raw_output_save_path,post_fix=post_fix)
    # df_occupancy = proc.get_occupancy(df_requests)
    list_df_charging_load = []
    # for i in ['pricing_double', 'pricing_double_BL_PV']:
    #     list_df_charging_load.append(proc.get_load_curve(raw_output_save_path=raw_output_save_path,sim_start_date=sim_start_date,post_fix=post_fix, agent_name=i))
    df_charging_load = proc.get_load_curve(
        raw_output_save_path=raw_output_save_path,
        sim_start_date=sim_start_date,
        post_fix=post_fix,
    )
    df_storage_load = proc.get_storage_load(
        raw_output_save_path=raw_output_save_path,
        sim_start_date=sim_start_date,
        post_fix=post_fix,
    )
    df_prices = proc.get_historical_prices(
        raw_output_save_path=raw_output_save_path,
        sim_start_date=sim_start_date,
        post_fix=post_fix,
    )
    df_episode_rewards = proc.get_episode_rewards(
        raw_output_save_path=raw_output_save_path,
        sim_start_date=sim_start_date,
        post_fix=post_fix,
    )

    # get combined load df
    df_combinded_loads = df_charging_load.merge(
        df_building_load, on="sim_time", how="left", suffixes=(None, "_x")
    )
    df_combinded_loads = df_combinded_loads.merge(
        df_storage_load, on="sim_time", how="left", suffixes=(None, "_x")
    )
    df_combinded_loads["combined_total_load"] = (
        df_combinded_loads["total_load"]
        + df_combinded_loads["load_kw_rescaled"]
        + df_combinded_loads["charge_load"]
    )
    df_combinded_loads = df_combinded_loads.merge(
        df_PV_generation, on="sim_time", how="left", suffixes=(None, "_x")
    )
    df_combinded_loads["grid_supply"] = (
        df_combinded_loads["combined_total_load"]
        - df_combinded_loads["pv_generation"]
        - df_combinded_loads["discharge_load"]
    )

    ####### execute plotting routines sequentially

    #####preferences (i.e, inputs)
    # plot_general_population_properties(request_data=df_requests,palette=palette,visuals_save_path=visuals_save_path, post_fix=post_fix )
    # plot_arrival_departure(request_data=df_requests, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix, full_version=full_version)
    # plot_combined_flexibility_indicators(request_data=df_requests, palette=palette, full_version=full_version, visuals_save_path=visuals_save_path, post_fix=post_fix)

    # try:
    #    plot_occupancy(occupancy_data=df_occupancy, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)
    # except:
    #    pass

    # COMMENTED OUT FOR PERFORMANCE
    # plot_avg_daily_occupancy_by_tech(occupancy_data=df_occupancy, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)
    # plot_avg_daily_occupancy_by_parker(occupancy_data=df_occupancy, palette=palette, out_path=visuals_save_path, post_fix=post_fix)

    #####OPERATIONS

    # load curve
    plot_full_sim_horizon_combined_load_sink_curve(
        charging_load_data=df_charging_load,
        building_load_data=df_building_load,
        storage_load_data=df_storage_load,
        palette=palette,
        visuals_save_path=visuals_save_path,
        post_fix=post_fix,
    )
    # plot_full_sim_horizon_combined_load_sink_curve_all_pricing(charging_load_data=list_df_charging_load, building_load_data=df_building_load, storage_load_data=df_storage_load, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)
    # plot_full_sim_horizon_combined_load_sink_curve_high_peak_comparison(charging_load_data=list_df_charging_load, building_load_data=df_building_load, storage_load_data=df_storage_load, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)
    plot_full_sim_horizon_combined_load_source_curve(
        charging_load_data=df_charging_load,
        building_load_data=df_building_load,
        pv_generation_data=df_PV_generation,
        grid_load_data=df_combinded_loads,
        post_fix=post_fix,
    )
    # plot_full_sim_horizon_charging_load_source_curve(charging_load_data=df_charging_load, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)

    # avg load curves (long execution)
    # plot_avg_charging_load_curves(charging_load_data=df_charging_load, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)
    # plot_avg_load_source_and_sink_curves(combined_load_data=df_combinded_loads)

    # cs use
    # df_charging_status, df_connection_status, df_energy_status, df_cs_utilization = proc.get_CS_utilization(raw_output_save_path=raw_output_save_path, post_fix=post_fix)
    # plot_cs_occupancy_profile(df_charging_status=df_charging_status, df_connection_status=df_connection_status, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)
    # plot_cs_utilization_energy(df_cs_utilization=df_cs_utilization, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)

    # service level
    # plot_service_level(request_data=df_requests, palette=palette, visuals_save_path=visuals_save_path, post_fix=post_fix)

    # historical_prices
    plot_historical_prices(
        prices=df_prices,
        palette=palette,
        visuals_save_path=visuals_save_path,
        post_fix=post_fix,
    )
    #
    plot_episodes_cumulative_rewards(
        df_episode_rewards=df_episode_rewards,
        palette=palette,
        visuals_save_path=visuals_save_path,
        post_fix="",
    )


#######################################
###### Plot Preference Data (i.e., sim input)
#######################################


def plot_general_population_properties(
    request_data,
    palette="mako",
    visuals_save_path="./Utilities/visuals_output/",
    post_fix="",
):
    """
    Plots user composition
    :param data:
    :return:
    """

    set_plotting_style(palette)

    # avg daily visitors
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 3])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])
    # colors = sns.color_palette(palette, n_colors=9)

    # share of parker types
    # prepare data
    request_daily = request_data.groupby(["day"], as_index=False).agg(
        {"arrival_time": "count"}
    )
    request_daily.columns = ["day", "num_visitors_total"]

    daily_visitors_avg = request_daily["num_visitors_total"].mean()
    daily_visitors_std = request_daily["num_visitors_total"].std()

    request_daily_ev = request_data.groupby(["day", "ev_yn"], as_index=False).agg(
        {"arrival_time": "count"}
    )
    request_daily_ev.columns = ["day", "ev", "num_visitors"]
    request_daily_ev = request_daily_ev.merge(request_daily, on="day", how="left")
    request_daily_ev["share"] = (
        request_daily_ev["num_visitors"] / request_daily_ev["num_visitors_total"]
    )

    request_daily_user_type = request_data.groupby(
        ["day", "user_type"], as_index=False
    ).agg({"arrival_time": "count"})
    request_daily_user_type.columns = ["day", "user_type", "num_visitors"]
    request_daily_user_type = request_daily_user_type.merge(
        request_daily, on="day", how="left"
    )
    request_daily_user_type["share"] = (
        request_daily_user_type["num_visitors"]
        / request_daily_user_type["num_visitors_total"]
    )

    # EV Y/N
    sns.barplot(
        y="share",
        x="ev",
        data=request_daily_ev,
        label="Avg Visitors",
        palette=palette,
        ax=ax0,
    )
    ax0.set_ylabel("")
    ax0.set_ylabel("Share in Total Daily Visits")
    ax0.set_title("Vehicle Type")
    ax0.set_xlabel("")
    ax0.set_ylim((0, 1))
    try:
        ax0.set_xticklabels(["ICE", "EV"])
    except:
        pass

    # User Type
    order_list = [
        "Business",
        "Morning_short",
        "Afternoon_short",
        "Evening_short",
        "Overnight",
        "Long-term",
    ]
    sns.barplot(
        y="share",
        x="user_type",
        data=request_daily_user_type,
        palette=palette,
        ax=ax1,
        order=order_list,
    )
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax1.set_title("Parker Type")
    ax1.set_ylim((0, 1))
    ax1.set_yticklabels([])
    ax1.set_xticklabels(
        [
            "Business",
            "Morning\n (short)",
            "Afternoon\n (short)",
            "Evening\n (short)",
            "Overnight",
            "Long-term",
        ]
    )

    fig.suptitle(
        "User Composition (Total Daily Visits: $\mu = ${}, $\sigma = ${})".format(
            round(daily_visitors_avg), round(daily_visitors_std)
        ),
        fontsize=14,
    )

    plt.tight_layout()
    plt.savefig(visuals_save_path + "general_population{}.pdf".format(post_fix))

    plt.show()


def plot_arrival_departure(
    request_data,
    palette="mako",
    visuals_save_path="./Utilities/visuals_output/",
    post_fix="",
    full_version=False,
):
    """
    Plots arrival and departure curves
    :return:
    """
    # style and colors
    palette = "mako",
    set_plotting_style(palette)
    colors = sns.color_palette(palette, n_colors=9)
    color_arrival = colors[2]
    color_departure = colors[-2]

    df = request_data

    if full_version:

        # OPTION 1:
        fig, ax = plt.subplots(figsize=(10, 4))
        sns.distplot(
            df["arrival_hour"],
            bins=24,
            hist=True,
            kde=False,
            norm_hist=True,
            ax=ax,
            label="Arrivals",
            color=color_arrival,
        )
        sns.distplot(
            df["departure_hour"],
            bins=24,
            hist=True,
            kde=False,
            norm_hist=True,
            ax=ax,
            label="Departures",
            color=color_departure,
        )

        ax.set_ylabel("Probability")
        ax.set_xlabel("Hour")
        ax.set_title("Distribution of Vehicle Arrival and Departure Times")
        ax.legend(loc="upper right", frameon=False)

        plt.tight_layout()
        plt.savefig(
            visuals_save_path
            + "arrival_departure_distributions_combined{}.pdf".format(post_fix)
        )

        plt.show()

    # OPTION 2:
    # prep data
    arrival_out = df.groupby(["day", "arrival_hour"], as_index=False).agg(
        {"arrival_time": "count"}
    )
    arrival_out.columns = ["day", "hour", "arrivals"]
    daily_arrivals = arrival_out.groupby(["day"], as_index=False).agg(
        {"arrivals": "sum"}
    )
    daily_arrivals.columns = ["day", "daily_arrivals"]
    arrival_out = arrival_out.merge(daily_arrivals, on="day", how="left")
    arrival_out["probability"] = arrival_out["arrivals"] / arrival_out["daily_arrivals"]

    departure_out = df.groupby(["day", "departure_hour"], as_index=False).agg(
        {"departure_time": "count"}
    )
    departure_out.columns = ["day", "hour", "departures"]
    daily_depart = departure_out.groupby(["day"], as_index=False).agg(
        {"departures": "sum"}
    )
    daily_depart.columns = ["day", "daily_depart"]
    departure_out = departure_out.merge(daily_depart, on="day", how="left")
    departure_out["probability"] = (
        departure_out["departures"] / departure_out["daily_depart"]
    ) * (-1)

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))
    sns.barplot(
        x="hour",
        y="probability",
        data=arrival_out,
        ax=ax,
        color=color_arrival,
        label="Arrivals",
    )
    sns.barplot(
        x="hour",
        y="probability",
        data=departure_out,
        ax=ax,
        color=color_departure,
        label="Departures",
    )

    ax.set_ylabel("Probability")
    ax.set_xlabel("Hour")
    ax.set_title("Distribution of Vehicle Arrival and Departure Times")
    ax.legend(loc="upper right", frameon=False)
    ax.yaxis.set_major_formatter(major_formatter)

    plt.tight_layout()
    plt.savefig(
        visuals_save_path
        + "arrival_departure_distributions_separate{}.pdf".format(post_fix)
    )

    plt.show()


def plot_occupancy(
    occupancy_data,
    palette="mako",
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):
    """
    Plots occupancy over sim period (total, by technology, by parker type)
    :return:
    """
    set_plotting_style()

    # OPTION 1: Over full sim period

    colors = sns.color_palette(palette, n_colors=6)

    fig, axes = plt.subplots(1, 2, figsize=(20, 8), sharey=True)

    # Plot 1: occupancy EVs by user type
    i = 0
    plot_data = occupancy_data.groupby(
        ["facility", "ev_yn", "time"], as_index=False
    ).agg({"total_occupancy": "sum"})
    x = np.array(plot_data[plot_data["ev_yn"] == 0]["time"])
    y2 = np.array(plot_data[plot_data["ev_yn"] == 0]["total_occupancy"])
    y1 = np.array(plot_data[plot_data["ev_yn"] == 1]["total_occupancy"])

    labels = ["EV", "ICE"]

    axes[i].stackplot(
        x, y1, y2, labels=labels, alpha=0.75, colors=[colors[-1], colors[0]]
    )
    axes[i].legend(loc="upper left", fontsize=12)
    axes[i].set_ylabel("# vehicles", fontsize=16)
    axes[i].set_xlabel("Simulation Time", fontsize=16)
    axes[i].set_title("Total Occupancy by Vehicle Technology", fontsize=16)

    # Plot 2: occupancy by user type (EVs only)
    i = 1
    plot_data = occupancy_data[(occupancy_data["ev_yn"] == 1)]

    # get arrays of data series
    x = np.array(plot_data[plot_data["user_type"] == "Business"]["time"])
    y1 = np.array(plot_data[plot_data["user_type"] == "Business"]["total_occupancy"])
    y2 = np.array(
        plot_data[plot_data["user_type"] == "Morning_short"]["total_occupancy"]
    )
    y3 = np.array(
        plot_data[plot_data["user_type"] == "Afternoon_short"]["total_occupancy"]
    )
    y4 = np.array(
        plot_data[plot_data["user_type"] == "Evening_short"]["total_occupancy"]
    )
    y5 = np.array(plot_data[plot_data["user_type"] == "Overnight"]["total_occupancy"])
    y6 = np.array(plot_data[plot_data["user_type"] == "Long-term"]["total_occupancy"])

    labels = [
        "Business",
        "Morning_short",
        "Afternoon_short",
        "Evening_short",
        "Overnight",
        "Long-term",
    ]

    axes[i].stackplot(
        x, y1, y2, y3, y4, y5, y6, labels=labels, alpha=0.75, colors=colors
    )
    axes[i].legend(loc="upper left", fontsize=12)
    axes[i].set_xlabel("Simulation Time", fontsize=16)
    axes[i].set_title("Total Occupancy by Parker Type (EVs only)", fontsize=16)

    plt.tight_layout()
    plt.savefig(visuals_save_path + "occupancy_full_sim{}.pdf".format(post_fix))

    plt.show()


def plot_avg_daily_occupancy_by_tech(
    occupancy_data,
    palette="mako",
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):
    """
    Plots daily occupancy curve (with conf. bands) per each parker type
    :param occupancy_data:
    :param palette:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    plot_data = occupancy_data.groupby(
        ["facility", "ev_yn", "time"], as_index=False
    ).agg({"total_occupancy": "sum"})

    plot_data["minutes_from_midnight"] = plot_data["time"].apply(
        lambda x: proc.minutes_from_midnight(x)
    )
    plot_data["hour"] = plot_data["minutes_from_midnight"] / 60

    labels = ["ICE", "EV"]

    sns.lineplot(
        x="hour",
        y="total_occupancy",
        data=plot_data,
        hue="ev_yn",
        ax=ax,
        palette=palette,
    )
    ax.legend(loc="upper left", fontsize=12, labels=labels)
    ax.set_ylabel("# vehicles", fontsize=14)
    ax.set_xlabel("Time of Day", fontsize=14)
    ax.set_title("Avg. Daily Occupancy Profile by Vehicle Technology", fontsize=14)

    plt.tight_layout()
    plt.savefig(visuals_save_path + "occupancy_dailyavg_by_tech{}.pdf".format(post_fix))

    plt.show()


def plot_avg_daily_occupancy_by_parker(
    occupancy_data, palette="mako", out_path="Results/visuals_output/", post_fix=""
):
    """
    Plots daily occupancy curve (with conf. bands) per each parker type
    :param occupancy_data:
    :param palette:
    :param out_path:
    :param post_fix:
    :return:
    """

    fig, ax = plt.subplots(figsize=(10, 4))

    plot_data = occupancy_data[(occupancy_data["ev_yn"] == 1)]

    plot_data["hour"] = plot_data["minutes_from_midnight"] / 60

    sns.lineplot(
        x="hour",
        y="total_occupancy",
        data=plot_data,
        hue="user_type",
        ax=ax,
        palette=palette,
    )
    ax.legend(loc="upper left", fontsize=12)
    ax.set_ylabel("# vehicles", fontsize=14)
    ax.set_xlabel("Time of Day", fontsize=14)
    ax.set_title("Avg. Daily Occupancy Profile by Parker Type (EV only)", fontsize=14)

    plt.tight_layout()
    plt.savefig(out_path + "occupancy_dailyavg_by_parker{}.pdf".format(post_fix))

    plt.show()


def plot_combined_flexibility_indicators(
    request_data,
    palette="mako",
    full_version=False,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):
    """

    :param request_data:
    :param palette:
    :param hue:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """
    set_plotting_style(palette)

    if full_version:
        # plots individual plots
        plot_energy_request_distribution(
            request_data=request_data,
            palette=palette,
            visuals_save_path=visuals_save_path,
            post_fix=post_fix,
        )
        plot_stay_duration_distribution(
            request_data=request_data,
            palette=palette,
            visuals_save_path=visuals_save_path,
            post_fix=post_fix,
        )
        plot_laxity_distribution(
            request_data=request_data,
            palette=palette,
            visuals_save_path=visuals_save_path,
            post_fix=post_fix,
        )

    # get combined plot

    data = request_data[request_data["ev_yn"] == 1]

    avg_park_duration = data["park_duration_h"].mean()
    avg_energy = data["energy_requested"].mean()
    avg_lax = data["laxity"].mean()

    colors = sns.color_palette(palette, n_colors=5)
    fig, axes = plt.subplots(1, 3, figsize=(10, 4), sharey=True)

    # park duration
    ax = axes[0]
    sns.histplot(
        x="park_duration_h",
        data=data,
        stat="probability",
        kde=False,
        color=colors[0],
        alpha=1,
        bins=25,
        ax=ax,
    )
    ax.set_xlabel("kWh")
    ax.set_xlim((-0.51, 36))
    ax.set_xlabel("Park Duration [h]")

    avg = avg_park_duration
    ax.vlines(x=avg, ymin=0, ymax=0.425, colors="k", linestyles="dashed")
    ax.text(avg + 1, 0.4, "$\mu =${}".format(round(avg, 2)))

    # energy
    ax = axes[1]
    sns.histplot(
        x="energy_requested",
        data=data,
        stat="probability",
        kde=False,
        color=colors[1],
        alpha=1,
        bins=25,
        ax=ax,
    )
    ax.set_xlabel("kWh")
    ax.set_xlim((-0.51, 60))
    ax.set_xlabel("Requested Energy [kWh]")

    avg = avg_energy
    ax.vlines(x=avg, ymin=0, ymax=0.425, colors="k", linestyles="dashed")
    ax.text(avg + 2, 0.4, "$\mu =${}".format(round(avg, 2)))

    # laxity

    ax = axes[2]
    sns.histplot(
        x="laxity",
        data=data,
        stat="probability",
        kde=False,
        color=colors[-1],
        alpha=1,
        bins=25,
        ax=ax,
    )
    ax.set_xlabel("kWh")
    ax.set_xlim((-0.51, 36))
    ax.set_xlabel("Laxity [h]")

    avg = avg_lax
    ax.vlines(x=avg, ymin=0, ymax=0.425, colors="k", linestyles="dashed")
    ax.text(avg + 1, 0.4, "$\mu =${}".format(round(avg, 2)))

    fig.suptitle("Population Flexibility Characteristics", fontsize=14)

    plt.tight_layout()
    plt.savefig(
        visuals_save_path
        + "flexibility_characteristics_combined{}.pdf".format(post_fix)
    )

    plt.show()


def plot_energy_request_distribution(
    request_data,
    palette="mako",
    hue=None,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):

    set_plotting_style(palette)
    data = request_data[request_data["ev_yn"] == 1]
    avg = data["energy_requested"].mean()
    colors = sns.color_palette(palette, n_colors=3)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax = sns.histplot(
        x="energy_requested",
        data=data,
        stat="probability",
        hue=hue,
        kde=False,
        color=colors[-1],
        alpha=1,
    )
    ax.set_xlabel("kWh")
    ax.set_xlim((-0.51, 100))
    ax.set_title("Empirical Distribution of Requested Energy", fontsize=14)

    # define avg
    ax.vlines(
        x=avg, ymin=0, ymax=ax.get_ylim()[1] - 0.025, colors="k", linestyles="dashed"
    )
    ax.text(avg + 1, ax.get_ylim()[1] - 0.05, "$\mu =${}".format(round(avg, 2)))

    plt.tight_layout()
    plt.savefig(
        visuals_save_path
        + "energy_request_distributions_separate{}.pdf".format(post_fix)
    )

    plt.show()


def plot_stay_duration_distribution(
    request_data,
    palette="mako",
    hue=None,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):
    """
    xxx
    :param request_data:
    :param palette:
    :param hue:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """

    set_plotting_style(palette)

    # process data
    data = request_data[request_data["ev_yn"] == 1]
    avg = data["park_duration_min"].mean()

    colors = sns.color_palette(palette, n_colors=5)

    # plot
    fig, ax = plt.subplots(figsize=(10, 4))

    ax = sns.histplot(
        x="park_duration_min",
        data=data,
        stat="probability",
        hue=hue,
        kde=False,
        color=colors[0],
        alpha=1,
    )
    ax.set_xlabel("Minutes")
    ax.set_xlim((-5, 1500))
    ax.set_title("Empirical Distribution of Stay Duration", fontsize=14)

    # define avg
    ax.vlines(
        x=avg, ymin=0, ymax=ax.get_ylim()[1] - 0.025, colors="k", linestyles="dashed"
    )
    ax.text(avg + 10, ax.get_ylim()[1] - 0.05, "$\mu =${}".format(round(avg, 2)))

    plt.tight_layout()
    plt.savefig(
        visuals_save_path
        + "stay_duration_distributions_separate{}.pdf".format(post_fix)
    )

    plt.show()


def plot_laxity_distribution(
    request_data,
    palette="mako",
    hue=None,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):
    """
    Plots laxity distribution
    :param request_data:
    :param palette:
    :param hue:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """

    set_plotting_style(palette)

    # select and process data
    data = request_data[request_data["ev_yn"] == 1]

    avg = data["laxity"].mean()

    colors = sns.color_palette(palette, n_colors=3)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax = sns.histplot(
        x="laxity",
        data=data,
        stat="probability",
        hue=hue,
        kde=False,
        color=colors[-1],
        alpha=1,
    )
    ax.set_xlabel("h")
    ax.set_xlim((-0.51, 24))
    ax.set_title("Empirical Distribution of Laxity", fontsize=14)

    # define avg
    ax.vlines(
        x=avg, ymin=0, ymax=ax.get_ylim()[1] - 0.025, colors="k", linestyles="dashed"
    )
    ax.text(avg + 1, ax.get_ylim()[1] - 0.05, "$\mu =${}".format(round(avg, 2)))

    plt.tight_layout()
    plt.savefig(
        visuals_save_path
        + "vehicle_laxity_distributions_separate{}.pdf".format(post_fix)
    )
    plt.show()


#######################################
###### Plot Operational KPIs
#######################################


def plot_full_sim_horizon_combined_load_sink_curve(
    charging_load_data,
    building_load_data,
    storage_load_data,
    palette="mako",
    hue=None,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
    agent_name=Configuration.instance().pricing_agent_name,
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

    data_charging = charging_load_data
    data_building = building_load_data
    data_storage = storage_load_data
    data_total = data_charging.merge(data_building, on="sim_time", how="left")
    data_total = data_total.merge(data_storage, on="sim_time", how="left")
    data_total["combined_total_load"] = (
        data_total["total_load"]
        + data_total["load_kw_rescaled"]
        + data_total["charge_load"]
    )

    data_charging.to_csv(
        visuals_save_path + agent_name + "_charging_data.csv", index=False
    )
    data_building.to_csv(
        visuals_save_path + agent_name + "_data_building.csv", index=False
    )
    data_storage.to_csv(
        visuals_save_path + agent_name + "_data_storage.csv", index=False
    )
    data_total.to_csv(visuals_save_path + agent_name + "_data_total.csv", index=False)

    max_load = data_total["combined_total_load"].max()
    xmax = data_total["sim_time"].max()

    colors = sns.color_palette(palette, n_colors=8)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(data_charging["total_load"], color=colors[0], label="EV Load")
    ax.plot(data_building["load_kw_rescaled"], color=colors[2], label="Building Load")
    ax.plot(data_storage["charge_load"], color=colors[4], label="Storage Load")
    ax.plot(data_total["combined_total_load"], color=colors[6], label="Combined")
    ax.set_xlabel("Simulation Time")
    ax.set_ylabel("Power [kW]")
    # ax.set_ylim(0, 950)
    ax.set_xlim((0 - 50, xmax + 50))
    ax.set_title(f"Electric loads using dynamic capacity-based pricing", fontsize=14)
    ax.legend(loc="upper right", ncol=2, fontsize=10)

    # define avg
    ax.hlines(
        y=max_load,
        xmin=0,
        xmax=xmax,
        colors="k",
        linestyles="dashed",
        label="Maximum Load = {}".format(round(max_load, 2)),
    )
    ax.text(
        0,
        max_load - 100,
        "Max. Combined Load: \n {} kW".format(round(max_load, 2)),
        fontsize=10,
    )

    plt.tight_layout()

    plt.savefig(
        visuals_save_path
        + f"full_sim_period_combined_load_sink_curve_{Configuration.instance().pricing_agent_name}.pdf"
    )

    plt.show()


def plot_full_sim_horizon_combined_load_sink_curve_high_peak_comparison(
    charging_load_data,
    building_load_data,
    storage_load_data,
    palette="mako",
    hue=None,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
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

    colors = sns.color_palette(palette, n_colors=8)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 3))
    titles = ["Low Peak Cost", "High Peak Cost"]
    counter = 0

    for i in range(2):
        data_charging = charging_load_data[counter]
        data_building = building_load_data
        data_storage = storage_load_data
        data_total = data_charging.merge(data_building, on="sim_time", how="left")
        data_total = data_total.merge(data_storage, on="sim_time", how="left")
        data_total["combined_total_load"] = (
            data_total["total_load"]
            + data_total["load_kw_rescaled"]
            + data_total["charge_load"]
        )

        max_load = data_total["combined_total_load"].max()
        xmax = data_total["sim_time"].max()
        ax[i].plot(data_charging["total_load"], color=colors[0], label="EV Load")
        ax[i].plot(data_total["combined_total_load"], color=colors[6], label="Combined")
        ax[i].set_xlabel("Simulation Time")
        ax[i].set_ylabel("Power [kW]")
        ax[i].set_ylim(0, 950)
        ax[i].set_xlim((0 - 50, xmax + 50))
        ax[i].set_title(f"Electric Loads {titles[counter]}", fontsize=14)
        ax[i].legend(loc="upper right", ncol=2, fontsize=10)

        # define avg
        ax[i].hlines(
            y=max_load,
            xmin=0,
            xmax=xmax,
            colors="k",
            linestyles="dashed",
            label="Maximum Load = {}".format(round(max_load, 2)),
        )
        ax[i].text(
            0,
            max_load - 100,
            "Max. Combined Load: \n {} kW".format(round(max_load, 2)),
            fontsize=10,
        )
        counter += 1

    plt.tight_layout()

    plt.savefig(visuals_save_path + f"high_peak_comparison.pdf")

    plt.show()


def plot_full_sim_horizon_combined_load_sink_curve_all_pricing(
    charging_load_data,
    building_load_data,
    storage_load_data,
    palette="mako",
    hue=None,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
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

    colors = sns.color_palette(palette, n_colors=8)
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    titles = [
        "Dynamic Multi-Power",
        "Static Multi-Power",
        "Dynamic Single-Power(22kW)",
        "Dynamic Single-Power(11kW)",
    ]
    counter = 0
    for i in range(2):

        data_charging = charging_load_data[counter]
        data_building = building_load_data
        data_storage = storage_load_data
        data_total = data_charging.merge(data_building, on="sim_time", how="left")
        data_total = data_total.merge(data_storage, on="sim_time", how="left")
        data_total["combined_total_load"] = (
            data_total["total_load"]
            + data_total["load_kw_rescaled"]
            + data_total["charge_load"]
        )

        max_load = data_total["combined_total_load"].max()
        xmax = data_total["sim_time"].max()
        ax[i].plot(data_charging["total_load"], color=colors[0], label="EV Load")
        ax[i].plot(data_total["combined_total_load"], color=colors[6], label="Combined")
        ax[i].set_xlabel("Simulation Time")
        ax[i].set_ylabel("Power [kW]")
        ax[i].set_ylim(0, 950)
        ax[i].set_xlim((0 - 50, xmax + 50))
        ax[i].set_title(f"Electric Loads {titles[counter]}", fontsize=14)
        ax[i].legend(loc="upper right", ncol=2, fontsize=10)

        # define avg
        ax[i].hlines(
            y=max_load,
            xmin=0,
            xmax=xmax,
            colors="k",
            linestyles="dashed",
            label="Maximum Load = {}".format(round(max_load, 2)),
        )
        ax[i].text(
            0,
            max_load - 100,
            "Max. Combined Load: \n {} kW".format(round(max_load, 2)),
            fontsize=10,
        )
        counter += 1

    plt.tight_layout()

    plt.savefig(visuals_save_path + f"benchmark_load_curve.pdf")

    plt.show()


def plot_full_sim_horizon_charging_load_source_curve(
    charging_load_data,
    palette="mako",
    hue=None,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
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

    data = charging_load_data

    max_load = data["total_load"].max()
    xmax = data["sim_time"].max()

    colors = sns.color_palette(palette, n_colors=5)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(data["total_load"])
    ax.set_xlabel("Simulation Time")
    ax.set_ylabel("Power [kW]")
    ax.set_xlim((0 - 50, xmax + 50))
    ax.set_title("Charging Load over Simulation Horizon", fontsize=14)

    # define avg
    ax.hlines(
        y=max_load,
        xmin=0,
        xmax=xmax,
        colors="k",
        linestyles="dashed",
        label="Maximum Load = {}".format(round(max_load, 2)),
    )
    ax.text(
        data["sim_time"].max() - 350,
        max_load - 250,
        "Max. Load: \n {} kW".format(round(max_load, 2)),
    )

    plt.tight_layout()

    plt.savefig(visuals_save_path + "full_sim_period_load_curve{}.pdf".format(post_fix))

    plt.show()


def plot_full_sim_horizon_combined_load_source_curve(
    charging_load_data,
    building_load_data,
    pv_generation_data,
    grid_load_data,
    palette="mako",
    hue=None,
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
    agent_name=Configuration.instance().pricing_agent_name,
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

    # combined_df = charging_load_data.merge(building_load_data, on="sim_time", how="left")
    # combined_df["combined_total_load"] = combined_df["total_load"] + combined_df["load_kw_rescaled"]
    # combined_df = combined_df.merge(pv_generation_data, on="sim_time",how="left")
    # combined_df["grid_supply"] = combined_df["combined_total_load"]-combined_df["pv_generation"]

    combined_df = grid_load_data
    combined_df.to_csv(visuals_save_path + agent_name + "_supply.csv", index=False)

    set_plotting_style()

    max_grid_load = combined_df["grid_supply"].max()
    xmax = combined_df["sim_time"].max()

    colors = sns.color_palette(palette, n_colors=8)
    fig, ax = plt.subplots(figsize=(10, 4))

    ax.plot(combined_df["pv_generation"], color=colors[0], label="PV Generation")
    ax.plot(combined_df["discharge_load"], color=colors[2], label="Storage Discharge")
    ax.plot(combined_df["grid_supply"], color=colors[4], label="Grid Supply")
    ax.plot(combined_df["combined_total_load"], color=colors[6], label="Total Supply")
    ax.set_xlabel("Simulation Time")
    ax.set_ylabel("Power [kW]")
    ax.set_xlim((0 - 50, xmax + 50))
    ax.set_title("Power Sources over Simulation Horizon", fontsize=14)
    ax.legend(loc="upper right", ncol=2, fontsize=10)

    # define avg
    ax.hlines(
        y=max_grid_load,
        xmin=0,
        xmax=xmax,
        colors="k",
        linestyles="dashed",
        label="Maximum Grid Load = {}".format(round(max_grid_load, 2)),
    )
    ax.text(
        0,
        max_grid_load - 100,
        "Max. Grid Load: \n {} kW".format(round(max_grid_load, 2)),
        fontsize=10,
    )

    plt.tight_layout()

    plt.savefig(
        visuals_save_path
        + "full_sim_period_combined_load_supply_curve{}.pdf".format(post_fix)
    )

    plt.show()


# plot avg daily load curve (box plots, differentiate by weekday, weekend)
def plot_avg_charging_load_curves(
    charging_load_data,
    palette="mako",
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):
    """
    Plots avg daily load curve (total, by day of week, by weekend vs weekday)
    :param charging_load_data:
    :param palette:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """

    set_plotting_style()

    charging_data = charging_load_data

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5), sharey=True)

    # set to hourly
    charging_data["hours_from_midnight"] = (
        charging_data["minutes_from_midnight_base5"] / 60
    )

    sns.lineplot(
        data=charging_data,
        x="hours_from_midnight",
        y="total_load",
        ax=axes[0],
        hue=None,
        palette=palette,
    )
    sns.lineplot(
        data=charging_data,
        x="hours_from_midnight",
        y="total_load",
        ax=axes[1],
        hue="day_of_week_string",
        palette=palette,
    )
    sns.lineplot(
        data=charging_data,
        x="hours_from_midnight",
        y="total_load",
        ax=axes[2],
        hue="weekday_yn_string",
        palette=palette,
    )

    for ax in axes:
        ax.set_xlabel("Hour of Day")
        ax.set_xlim(0, 24)
    for ax in [axes[1], axes[2]]:
        ax.legend(loc="upper right", fontsize=10)

    axes[0].set_ylabel("Total Load from EV Charging [kW]")
    plt.suptitle("Avg. Daily Charging Load Curve", fontsize=14)  # , y=1.02

    plt.tight_layout()
    plt.savefig(visuals_save_path + "avg_load_curve{}.pdf".format(post_fix))

    plt.show()


def plot_avg_load_source_and_sink_curves(
    combined_load_data,
    palette="mako",
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):
    """
    Plots avg daily load curve (total, by day of week, by weekend vs weekday)
    :param charging_load_data:
    :param palette:
    :param visuals_save_path:
    :param post_fix:
    :return:
    """

    set_plotting_style()
    colors = sns.color_palette(palette, n_colors=9)

    combined_load_data["hours_from_midnight"] = (
        combined_load_data["minutes_from_midnight_base5"] / 60
    )
    combined_load_data["total_load_sources"] = (
        combined_load_data["pv_generation"]
        + combined_load_data["grid_supply"]
        + combined_load_data["discharge_load"]
    )
    combined_load_data["total_load_sinks"] = (
        combined_load_data["total_load"]
        + combined_load_data["load_kw_rescaled"]
        + combined_load_data["charge_load"]
    )

    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=True)

    # SINKS
    sns.lineplot(
        data=combined_load_data,
        x="hours_from_midnight",
        y="total_load",
        ax=axes[0],
        hue=None,
        palette=[colors[0]],
        label="EV Charging",
    )
    sns.lineplot(
        data=combined_load_data,
        x="hours_from_midnight",
        y="load_kw_rescaled",
        ax=axes[0],
        hue=None,
        palette=[colors[3]],
        label="Building Baseload",
    )
    sns.lineplot(
        data=combined_load_data,
        x="hours_from_midnight",
        y="charge_load",
        ax=axes[0],
        hue=None,
        palette=[colors[6]],
        label="Storage Charging",
    )
    sns.lineplot(
        data=combined_load_data,
        x="hours_from_midnight",
        y="total_load_sinks",
        ax=axes[0],
        hue=None,
        palette=[colors[8]],
        label="Total Demand",
    )

    # SOURCES
    sns.lineplot(
        data=combined_load_data,
        x="hours_from_midnight",
        y="pv_generation",
        ax=axes[1],
        hue=None,
        palette=[colors[0]],
        label="PV Generation",
    )
    sns.lineplot(
        data=combined_load_data,
        x="hours_from_midnight",
        y="grid_supply",
        ax=axes[1],
        hue=None,
        palette=[colors[3]],
        label="Grid Supply",
    )
    sns.lineplot(
        data=combined_load_data,
        x="hours_from_midnight",
        y="discharge_load",
        ax=axes[1],
        hue=None,
        palette=[colors[6]],
        label="Storage Discharging",
    )
    sns.lineplot(
        data=combined_load_data,
        x="hours_from_midnight",
        y="total_load_sources",
        ax=axes[1],
        hue=None,
        palette=[colors[8]],
        label="Total Supply",
    )

    for ax in axes:
        ax.set_xlabel("Hour of Day")
        ax.set_xlim(0, 24)
        ax.legend(loc="upper left", fontsize=10)

    axes[0].set_title("Load Usage")
    axes[0].set_ylabel("Load [kW]")
    axes[1].set_title("Load Supply")
    plt.suptitle("Avg. Daily Load Curve", fontsize=14)  # , y=1.02

    plt.tight_layout()
    plt.savefig(
        visuals_save_path + "avg_load_source_sink_curves{}.pdf".format(post_fix)
    )

    plt.show()


# plot occupancy level per charger
def plot_cs_occupancy_profile(
    df_charging_status,
    df_connection_status,
    palette="mako",
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):

    set_plotting_style()

    fig, axes = plt.subplots(2, 1, figsize=(16, 9), sharex=True)
    sns.heatmap(
        df_connection_status.drop("total_vehicle_connection_periods", axis=1),
        ax=axes[0],
        cmap=palette,
    )
    sns.heatmap(
        df_charging_status.drop(
            ["total_vehicle_charging_periods", "total_periods", "total_days"], axis=1
        ),
        ax=axes[1],
        cmap=palette,
    )

    axes[0].set_title(
        "Number of Connected Vehicles per Charger and Period", fontsize=14
    )
    axes[0].set_ylabel("Charger ID")

    axes[1].set_title("Number of Charging Vehicles per Charger and Period", fontsize=14)
    axes[1].set_xlabel("Simulation Time")
    axes[1].set_ylabel("Charger ID")

    plt.tight_layout()
    plt.savefig(visuals_save_path + "cs_occupancy_heat_profile{}.pdf".format(post_fix))

    plt.show()


# TODO!!!
# plot utilization per charger ((1) time used for charging, (3) vehicles served per day)


def plot_cs_utilization_energy(
    df_cs_utilization,
    palette="mako",
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):

    set_plotting_style()

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=False)

    sns.boxplot(
        data=df_cs_utilization,
        y="daily_energy_transfer",
        ax=axes[0],
        palette=palette,
        fliersize=1,
    )
    sns.boxplot(
        data=df_cs_utilization,
        y="utilization_energy_transfer",
        ax=axes[1],
        palette=palette,
        fliersize=1,
    )

    axes[0].set_ylabel("Energy Transfer per Charger (kWh/day)")
    axes[1].set_ylabel("Utilization (%)")

    plt.suptitle("Charger Utilization (Energy Transfer)", fontsize=14)
    plt.tight_layout()

    plt.savefig(visuals_save_path + "cs_utilization_energy{}.pdf".format(post_fix))

    plt.show()


# plot service level (check good visualization options)
def plot_service_level(
    request_data,
    palette="mako",
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):

    set_plotting_style()

    # avg daily visitors
    fig = plt.figure(figsize=(10, 4))
    gs = gridspec.GridSpec(1, 2, width_ratios=[1, 6])
    ax0 = plt.subplot(gs[0])
    ax1 = plt.subplot(gs[1])

    data = request_data[request_data["ev_yn"] == 1]

    avg_service_level = data["energy_charged"].sum() / data["energy_requested"].sum()

    order_list = [
        "Business",
        "Morning_short",
        "Afternoon_short",
        "Evening_short",
        "Overnight",
        "Long-term",
    ]
    sns.boxplot(
        x="user_type",
        y="service_level",
        data=data,
        palette=palette,
        order=order_list,
        fliersize=1,
        ax=ax1,
    )
    sns.boxplot(
        y="service_level",
        data=data,
        palette=palette,
        order=order_list,
        fliersize=1,
        ax=ax0,
    )

    # ax0.set_xlabel("All",fontsize = 12)
    ax0.set_ylabel("Service Level", fontsize=14)
    ax0.set_xticklabels(["Total"])

    # ax1.set_xlabel("User Type",fontsize = 12)
    # ax1.set_title("Service Level by User Type",fontsize = 14)
    ax1.set_ylabel("")
    ax1.set_xlabel("")
    ax1.set_yticks([])
    ax1.set_xticklabels(
        [
            "Business",
            "Morning\n (short)",
            "Afternoon\n (short)",
            "Evening\n (short)",
            "Overnight",
            "Long-term",
        ]
    )

    plt.suptitle(
        "Service Level ($\mu =${})".format(round(avg_service_level, 2)), fontsize=14
    )  # ,y=1.035

    plt.tight_layout()
    plt.savefig(visuals_save_path + "service_level{}.pdf".format(post_fix))

    plt.show()


# Plot historical prices
def plot_historical_prices(
    prices, palette="mako", visuals_save_path="Utilities/visuals_output/", post_fix=""
):
    set_plotting_style()

    data_pricing = prices
    data_total = data_pricing

    colors = sns.color_palette(palette, n_colors=8)
    fig, ax = plt.subplots(figsize=(10, 4))

    counter = 0
    print(data_pricing.columns)
    for x in ["0", "1"]:
        ax.plot(data_pricing[x], color=colors[counter], label=x)
        ax.set_xlabel("Simulation Time")
        ax.set_ylabel("Price [$]")
        ax.set_title("Energy Prices over Simulation Horizon", fontsize=14)
        ax.legend(loc="upper right", ncol=2, fontsize=10)
        counter += 1

    # define avg

    plt.tight_layout()

    plt.savefig(visuals_save_path + "full_sim_period_pricing{}.pdf".format(post_fix))

    plt.show()


def plot_episodes_cumulative_rewards(
    df_episode_rewards,
    palette="mako",
    visuals_save_path="Utilities/visuals_output/",
    post_fix="",
):
    fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(12, 6), constrained_layout=True)
    ax.plot(
        df_episode_rewards.index,
        df_episode_rewards["Dynamic-Cap-Pricing"],
        label="Dynamic-Cap-Pricing",
        linestyle="dotted",
        marker="x",
    )
    ax.plot(
        df_episode_rewards.index,
        df_episode_rewards["Time-of-Use"],
        label="Time-of-Use",
        marker="v",
    )
    # ax.plot(df_episode_rewards.index, df_episode_rewards['Perfect-Info'], label='Perfect-Info', linestyle='dotted', marker='o')
    # ax.plot(df_episode_rewards.index, df_episode_rewards['Baseline'], label='Baseline', linestyle='dotted',
    #         marker='o')
    ax.plot(
        df_episode_rewards.index,
        df_episode_rewards["Dynamic-Traditional"],
        label="Dynamic-Traditional",
        marker=".",
    )
    ax.plot(
        df_episode_rewards.index,
        df_episode_rewards["Dynamic-Menu-Based"],
        label="Dynamic-Menu-Based",
        linestyle="dotted",
        marker="o",
    )

    # ax.plot(profit_data['Episode'], profit_data['Limited_DQN'], label='Limited_MARL', linestyle='solid', marker='v')
    # ax.plot(profit_data['Episode'], profit_data['Limited_HDQN'], label='Limited_HMARL (MADC)', linestyle='solid', marker='x')

    ax.ticklabel_format(style="sci", scilimits=(-3, 4), axis="y")
    ax.set_xlabel("Episode", fontsize=12)
    ax.set_ylabel("EVCH Profits($)", fontsize=12)
    ax.set_title("Objective Function", fontsize=12)
    ax.legend(fontsize=11)
    ax.tick_params(axis="both", labelsize=12)
    # ax.set_ylim(0,)
    plt.tight_layout()

    plt.savefig(visuals_save_path + "episode_rewards{}.pdf".format(post_fix))

    plt.show()


#######################################
###### Plot Investment KPIs
#######################################


############ HELPERS
def minutes_from_midnight(dt):
    mfm = dt.hour * 60 + dt.minute
    return mfm


def round_down_to_base(num, base):
    """
    Rounds down to a given base.
    Ex. round_down_to_base (19,5) = 15; round_down_to_base (21,5) = 20
    :param num: number to be rounded down
    :param base: base against which to round down
    :return:
    """
    return num - (num % base)


@ticker.FuncFormatter
def major_formatter(x, pos):
    label = str(round(-x, 2)) if x < 0 else str(round(x, 2))
    return label
