import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import set_matplotlib_formats

set_matplotlib_formats("retina")
import zipfile


def set_style():
    # This sets reasonable defaults for font size for a figure that will go in a paper
    sns.set_context("paper")

    # Set the font to be serif, rather than sans
    sns.set(font="serif")
    # sns.set_color_codes("pastel")
    sns.set_palette("mako", n_colors=3)
    # Make the background white, and specify the font family
    sns.set_style(
        "ticks", {"font.family": "serif", "font.serif": ["Times", "Palatino", "serif"]}
    )


dates = [
    "2019-06-03",
    "2019-06-04",
    "2019-06-05",
    "2019-06-06",
    "2019-06-07",
    "2019-10-21",
    "2019-10-22",
    "2019-10-23",
    "2019-10-24",
    "2019-10-25",
]


def visualize(data=None, zip_name=None, type="CSs"):
    if data is None:
        data = dates
    if zip_name is None:
        zip_name = ["Facility_3", "Facility_4", "Facility_6"]
    set_style()

    if type == "Energy":
        vehicles_summer_3 = dict()
        dgs_energy_summer_3 = dict()
        df_energy_summer_3 = pd.DataFrame()

        vehicles_summer_4 = dict()
        dgs_energy_summer_4 = dict()
        df_energy_summer_4 = pd.DataFrame()

        vehicles_summer_6 = dict()
        dgs_energy_summer_6 = dict()
        df_energy_summer_6 = pd.DataFrame()
        zf_3 = zipfile.ZipFile(f"{zip_name[0]}.zip")
        zf_4 = zipfile.ZipFile(f"{zip_name[1]}.zip")
        zf_6 = zipfile.ZipFile(f"{zip_name[2]}.zip")
        facility_3 = "Facility_3"
        facility_4 = "Facility_4"
        facility_6 = "Facility_6"
        for i in range(0, 5):
            vehicles_summer_3[i] = pd.read_csv(
                zf_3.open(f"vehicles_{facility_3}_{data[i]}.csv")
            )
            vehicles_summer_4[i] = pd.read_csv(
                zf_4.open(f"vehicles_{facility_4}_{data[i]}.csv")
            )
            vehicles_summer_6[i] = pd.read_csv(
                zf_6.open(f"vehicles_{facility_6}_{data[i]}.csv")
            )

            vehicles_summer_3[i].loc[vehicles_summer_3[i]["Energy"] < 0] = 0
            vehicles_summer_4[i].loc[vehicles_summer_4[i]["Energy"] < 0] = 0
            vehicles_summer_6[i].loc[vehicles_summer_6[i]["Energy"] < 0] = 0

            dgs_energy_summer_3[i] = (
                vehicles_summer_3[i]
                .groupby(["time"])
                .agg({"Energy": "sum"})
                .reset_index()
            )
            dgs_energy_summer_3[i]["day"] = data[i]
            df_energy_summer_3 = df_energy_summer_3.append(dgs_energy_summer_3[i])

            dgs_energy_summer_4[i] = (
                vehicles_summer_4[i]
                .groupby(["time"])
                .agg({"Energy": "sum"})
                .reset_index()
            )
            dgs_energy_summer_4[i]["day"] = data[i]
            df_energy_summer_4 = df_energy_summer_4.append(dgs_energy_summer_4[i])

            dgs_energy_summer_6[i] = (
                vehicles_summer_6[i]
                .groupby(["time"])
                .agg({"Energy": "sum"})
                .reset_index()
            )
            dgs_energy_summer_6[i]["day"] = data[i]
            df_energy_summer_6 = df_energy_summer_6.append(dgs_energy_summer_6[i])

        dff = pd.DataFrame(columns=["time", "day", "a", "Energy"])
        # dff.loc[0] = [23, '2019-10-21', "Energy", 0]
        # dff.loc[0] = [23, '2019-10-22', "Energy", 0]
        # dff.loc[0] = [23, '2019-10-23', "Energy", 0]

        dg_energy_summer_3 = df_energy_summer_3.groupby(["time", "day"]).agg("mean")
        dg_energy_summer_3 = pd.DataFrame(dg_energy_summer_3.stack()).reset_index()
        dg_energy_summer_3.columns = ["time", "day", "a", "Energy"]
        dg_energy_summer_3.loc[(dg_energy_summer_3.time <= 5), "Energy"] = 0
        dg_energy_summer_3 = dg_energy_summer_3.append(dff)

        dg_energy_summer_4 = df_energy_summer_4.groupby(["time", "day"]).agg("mean")
        dg_energy_summer_4 = pd.DataFrame(dg_energy_summer_4.stack()).reset_index()
        dg_energy_summer_4.columns = ["time", "day", "a", "Energy"]
        dg_energy_summer_4.loc[(dg_energy_summer_4.time <= 5), "Energy"] = 0
        dg_energy_summer_4 = dg_energy_summer_4.append(dff)

        dg_energy_summer_6 = df_energy_summer_6.groupby(["time", "day"]).agg("mean")
        dg_energy_summer_6 = pd.DataFrame(dg_energy_summer_6.stack()).reset_index()
        dg_energy_summer_6.columns = ["time", "day", "a", "Energy"]
        dg_energy_summer_6.loc[(dg_energy_summer_6.time <= 5), "Energy"] = 0
        dg_energy_summer_6 = dg_energy_summer_6.append(dff)

        vehicles_winter_3 = dict()
        dgs_energy_winter_3 = dict()
        df_energy_winter_3 = pd.DataFrame()

        vehicles_winter_4 = dict()
        dgs_energy_winter_4 = dict()
        df_energy_winter_4 = pd.DataFrame()

        vehicles_winter_6 = dict()
        dgs_energy_winter_6 = dict()
        df_energy_winter_6 = pd.DataFrame()

        for i in range(5, 10):
            vehicles_winter_3[i] = pd.read_csv(
                zf_3.open(f"vehicles_{facility_3}_{data[i]}.csv")
            )
            vehicles_winter_4[i] = pd.read_csv(
                zf_4.open(f"vehicles_{facility_4}_{data[i]}.csv")
            )
            vehicles_winter_6[i] = pd.read_csv(
                zf_6.open(f"vehicles_{facility_6}_{data[i]}.csv")
            )

            vehicles_winter_3[i].loc[vehicles_winter_3[i]["Energy"] < 0] = 0
            vehicles_winter_4[i].loc[vehicles_winter_4[i]["Energy"] < 0] = 0
            vehicles_winter_6[i].loc[vehicles_winter_6[i]["Energy"] < 0] = 0

            dgs_energy_winter_3[i] = (
                vehicles_winter_3[i]
                .groupby(["time"])
                .agg({"Energy": "sum"})
                .reset_index()
            )
            dgs_energy_winter_3[i]["day"] = data[i]
            df_energy_winter_3 = df_energy_winter_3.append(dgs_energy_winter_3[i])

            dgs_energy_winter_4[i] = (
                vehicles_winter_4[i]
                .groupby(["time"])
                .agg({"Energy": "sum"})
                .reset_index()
            )
            dgs_energy_winter_4[i]["day"] = data[i]
            df_energy_winter_4 = df_energy_winter_4.append(dgs_energy_winter_4[i])

            dgs_energy_winter_6[i] = (
                vehicles_winter_6[i]
                .groupby(["time"])
                .agg({"Energy": "sum"})
                .reset_index()
            )
            dgs_energy_winter_6[i]["day"] = data[i]
            df_energy_winter_6 = df_energy_winter_6.append(dgs_energy_winter_6[i])

        dg_energy_winter_3 = df_energy_winter_3.groupby(["time", "day"]).agg("mean")
        dg_energy_winter_3 = pd.DataFrame(dg_energy_winter_3.stack()).reset_index()
        dg_energy_winter_3.columns = ["time", "day", "a", "Energy"]
        dg_energy_winter_3.loc[(dg_energy_winter_3.time <= 5), "Energy"] = 0
        dg_energy_winter_3 = dg_energy_winter_3.append(dff)

        dg_energy_winter_4 = df_energy_winter_4.groupby(["time", "day"]).agg("mean")
        dg_energy_winter_4 = pd.DataFrame(dg_energy_winter_4.stack()).reset_index()
        dg_energy_winter_4.columns = ["time", "day", "a", "Energy"]
        dg_energy_winter_4.loc[(dg_energy_winter_4.time <= 5), "Energy"] = 0
        dg_energy_winter_4 = dg_energy_winter_4.append(dff)

        dg_energy_winter_6 = df_energy_winter_6.groupby(["time", "day"]).agg("mean")
        dg_energy_winter_6 = pd.DataFrame(dg_energy_winter_6.stack()).reset_index()
        dg_energy_winter_6.columns = ["time", "day", "a", "Energy"]
        dg_energy_winter_6.loc[(dg_energy_winter_6.time <= 5), "Energy"] = 0
        dg_energy_winter_6 = dg_energy_winter_6.append(dff)
        print(dg_energy_winter_6)

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))

        ax[0].set_xlim((0, 24))
        ax[0].set_ylim((0, 1200))
        ax[0].set_xlabel("Time of day", fontsize=12)
        ax[0].set_ylabel("Energy (KWh)", fontsize=12)
        ax[0].set_title("Energy consumption in summer days", fontsize=12)

        ax[1].set_xlim((0, 24))
        ax[1].set_ylim((0, 1200))
        ax[1].set_xlabel("Time of day", fontsize=12)
        ax[1].set_ylabel("Energy (KWh)", fontsize=12)
        ax[1].set_title("Energy consumption in winter days", fontsize=12)

        sns.lineplot(
            data=dg_energy_summer_3,
            x="time",
            y="Energy",
            ci=95,
            legend=True,
            ax=ax[0],
            label="Mixed-use",
        )
        sns.lineplot(
            data=dg_energy_winter_3,
            x="time",
            y="Energy",
            ci=95,
            legend=True,
            ax=ax[1],
            label="Mixed-use",
        )
        sns.lineplot(
            data=dg_energy_summer_4,
            x="time",
            y="Energy",
            ci=95,
            legend=True,
            ax=ax[0],
            label="Destination",
        )
        sns.lineplot(
            data=dg_energy_winter_4,
            x="time",
            y="Energy",
            ci=95,
            legend=True,
            ax=ax[1],
            label="Destination",
        )
        sns.lineplot(
            data=dg_energy_summer_6,
            x="time",
            y="Energy",
            ci=95,
            legend=True,
            ax=ax[0],
            label="Workplace",
        )
        sns.lineplot(
            data=dg_energy_winter_6,
            x="time",
            y="Energy",
            ci=95,
            legend=True,
            ax=ax[1],
            label="Workplace",
        )
        """sns.boxplot(data=dg_energy_summer_3, x='time', y='Energy', ax=ax[0], label='Mixed-use')
        sns.boxplot(data=dg_energy_winter_3, x='time', y='Energy', ax=ax[1], label='Mixed-use')
        sns.boxplot(data=dg_energy_summer_4, x='time', y='Energy', ax=ax[0], label='Destination')
        sns.boxplot(data=dg_energy_winter_4, x='time', y='Energy', ax=ax[1], label='Destination')
        sns.boxplot(data=dg_energy_summer_6, x='time', y='Energy', ax=ax[0], label='Workplace')
        sns.boxplot(data=dg_energy_winter_6, x='time', y='Energy', ax=ax[1], label='Workplace')"""

        """sns.boxplot(data=dg_energy_summer, x='time', y='Energy', ax=ax[1][0])
        sns.boxplot(data=dg_energy_winter, x='time', y='Energy', ax=ax[1][1])"""
        plt.tight_layout()
        ax[0].legend(loc="upper right", fontsize=10)
        ax[1].legend(loc="upper right", fontsize=10)
        plt.savefig("Energy.pdf")
        plt.show()
        return dg_energy_winter_6

    if type == "CSs":
        costs = pd.read_csv(f"results.csv")

        fig, ax = plt.subplots(1, 2, figsize=(10, 4))
        """sns.boxplot(data=costs, x='Facility Title', y='NoC', hue='Season', ax=ax[0])
        sns.boxplot(data=costs, x='Facility Title', y='NoP', hue='Season', ax=ax[1])"""
        sns.boxplot(
            data=costs[costs["Season"] == "Summer"],
            x="Facility Title",
            y="NoC",
            ax=ax[0],
        )
        sns.boxplot(
            data=costs[costs["Season"] == "Summer"],
            x="Facility Title",
            y="P_plus",
            ax=ax[1],
        )
        # ax[0].set_xlim((0, 24))
        # ax[0].set_ylim((0, 1500))
        ax[0].set_xlabel("Facility Type", fontsize=12)
        ax[0].set_ylabel("Number", fontsize=12)
        ax[0].set_title("Number of EVSEs", fontsize=12)

        # ax[1].set_xlim((0, 24))
        # ax[1].set_ylim((300, 370))
        ax[1].set_xlabel("Facility Type", fontsize=12)
        ax[1].set_ylabel("Power(KW)", fontsize=12)
        ax[1].set_title("Grid expansion", fontsize=12)
        plt.tight_layout()
        plt.savefig("CSs.pdf")
        plt.show()

    if type == "Benchmark":
        costs = pd.read_csv(f"results_adoption.csv")
        costs["Solution"] = "Optimal"
        benchmark = pd.read_csv(f"results_benchmark.csv")
        benchmark["Solution"] = "Status quo"
        df = costs.append(benchmark)
        df["Installation_costs"] = df["Installation_costs"] * 365 * 2
        fig, ax = plt.subplots(2, 2, figsize=(16, 8))

        """sns.boxplot(data=costs, x='Facility Title', y='NoC', hue='Season', ax=ax[0])
        sns.boxplot(data=costs, x='Facility Title', y='NoP', hue='Season', ax=ax[1])"""
        sns.barplot(
            data=df[df["Season"] == "Summer"],
            x="Adoption",
            y="NoC",
            hue="Solution",
            ax=ax[0][0],
        )
        sns.barplot(
            data=df[df["Season"] == "Summer"],
            x="Adoption",
            y="P_plus",
            hue="Solution",
            ax=ax[0][1],
        )
        sns.barplot(
            data=df[df["Season"] == "Summer"],
            x="Adoption",
            y="Installation_costs",
            hue="Solution",
            ax=ax[1][0],
        )
        sns.barplot(
            data=df[df["Season"] == "Summer"],
            x="Adoption",
            y="Operations_costs",
            hue="Solution",
            ax=ax[1][1],
        )
        # ax[0].set_xlim((0, 24))
        # ax[0].set_ylim((0, 1500))
        ax[0][0].set_xlabel("EV adoption rate", fontsize=20)
        ax[0][0].set_ylabel("Number", fontsize=20)
        ax[0][0].set_title("Number of EVSEs", fontsize=20)

        # ax[1].set_xlim((0, 24))
        # ax[1].set_ylim((0, 1500))
        ax[0][1].set_xlabel("EV adoption rate", fontsize=20)
        ax[0][1].set_ylabel("Power (KW)", fontsize=20)
        ax[0][1].set_title("Grid expansion", fontsize=20)

        ax[1][0].set_xlabel("EV adoption rate", fontsize=20)
        ax[1][0].set_ylabel("Cost (USD)", fontsize=20)
        ax[1][0].set_title("Investment cost", fontsize=20)

        # ax[1].set_xlim((0, 24))
        # ax[1].set_ylim((0, 1500))
        ax[1][1].set_xlabel("EV adoption rate", fontsize=20)
        ax[1][1].set_ylabel("Cost (USD)", fontsize=20)
        ax[1][1].set_title("Daily operations cost", fontsize=20)
        # ax[0][0].legend(loc='upper left', fontsize=9)
        # ax[0][1].legend(loc='upper left', fontsize=9)
        # ax[1][1].legend(loc='upper left', fontsize=9)
        # ax[1][0].legend(loc='upper left', fontsize=9)
        plt.tight_layout()
        plt.savefig("benchmark.pdf")
        plt.show()

    if type == "Utilization":
        costs = pd.read_csv(f"results_plugs.csv")
        facility = "Facility_3"
        for i in range(0, 5):
            for p in [1, 2, 4, 6]:
                zf = zipfile.ZipFile(f"Facility_3_{p}plugs.zip")
                df = pd.read_csv(zf.open(f"vehicles_{facility}_{data[i]}.csv"))
                df.loc[df["Energy"] < 0] = 0
                costs.loc[
                    (costs["Date"] == data[i]) & (costs["Plugs"] == p), "Utilization"
                ] = df.groupby(["time"]).agg({"Energy": "sum"}).reset_index().mean()[
                    "Energy"
                ] / (
                    costs.loc[(costs["Date"] == data[i]) & (costs["Plugs"] == p), "NoC"]
                    * 22
                )
        costs["P_plus"] = costs["P_plus"] * 1.25
        df = costs
        fig, ax = plt.subplots(1, 2, figsize=(16, 8))
        sns.barplot(
            data=df[df["Season"] == "Summer"], x="Plugs", y="Utilization", ax=ax[0]
        )
        sns.boxplot(data=df[df["Season"] == "Summer"], x="Plugs", y="P_plus", ax=ax[1])

        ax[0].set_xlabel("Number of connectors per EVSE", fontsize=16)
        ax[0].set_ylabel("Percentage", fontsize=16)
        ax[0].set_title("Utilization", fontsize=16)

        ax[1].set_xlabel("Number of connectors per EVSE", fontsize=16)
        ax[1].set_ylim((375, 460))
        ax[1].set_ylabel("Power(KW)", fontsize=16)
        ax[1].set_title("Grid expansion", fontsize=16)

        plt.tight_layout()
        plt.savefig("Utilization.pdf")
        plt.show()


"""costs = pd.read_csv(f'costs.csv')
C_plug = 250 / 365 / 20
C_EVSE = 4500 / 365 / 20
C_grid = 240 / 365 / 20
T_p = 15.48 / 30"""

"""def invest_cost(a, b, c, d):
    return a * C_EVSE + b * C_plug + c * C_grid + d * T_p

costs['Installation_costs'] = costs.apply(
    lambda row: invest_cost(row['NoC'], row['NoP'], row['P_plus'], row['P_star']), axis=1)
costs['Operations_costs'] = costs.apply(
    lambda row: row['costs']-row['Operations_costs'], axis=1)"""
