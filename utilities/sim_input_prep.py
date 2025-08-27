import numpy as np
import pandas as pd
import seaborn as sns
from datetime import datetime
from matplotlib import pyplot as plt
from pandas.plotting import register_matplotlib_converters

register_matplotlib_converters()
from numpy.random import default_rng

# ML/Clustering stuff
# from sklearn.preprocessing import StandardScaler
# from sklearn.cluster import KMeans

import warnings

warnings.filterwarnings(action="ignore")

# CONFIG

DATA_PATH = "/home/rahadi/Projects/EVCC/kshroer/EV_Charging_Clusters/Data/"
OUT_PATH = ""


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


set_plotting_style()

# Preference Routine is as follows:
#
# 1. Arrival and Departure Patterns:
#     - We use APCOA Parking Data to determine stochastic arrival ($A$), stay duration ($\delta$) and departure ($D$) preferences
# 1. Request Charging Y/N:
#     - EV share ($\sigma_{EV}$): For each vehicle we select the propulsion technology probilistically based on a Brenoulli Process with $p(EV)=\sigma_{EV}$ and $p(ICE)=(1-\sigma_{EV})$
#     - Charge Y/N: Per each EV we decide whether the vehicle requires charging or not. We model this as probability distribution conditional on duration of stay $p(charge=True | \delta)$
# 1. Exact Charge Demand:
#     - Option I: Collaborative Filtering using ACN Data
#     - Option II: Simplified Transportation Demand Model, assuming daily commuting distances and home charging opportunity (Y/N)
#     - Option III: Assume start and end SOC as well as battery sizes (likely the inverior option)

# Load Data
parkings = pd.read_csv(
    DATA_PATH + "EV_Energy_Demand_Data/Parking+Charging_Data_BLENDED_CLUSTERED_v2.csv"
)
parkings = parkings[
    [
        "EntryDateTime",
        "ExitDateTime",
        "HoursStay",
        "MinutesStay",
        "RevenueAmount",
        "SiteID",
        "Year",
        "EntryMFM",
        "ExitMFM",
        "EntryDate",
        "ExitDate",
        "EntryHour",
        "ExitHour",
        "EntryDayOfWeek",
        "EntryWeekday_yn",
        "EntryHoliday_yn",
        "userInputs_kWhRequested",
        "ClusterName",
    ]
]  # 'ClusterName',

# limit parking duration to 48h (let's not do this for now!)
parkings = parkings[parkings["MinutesStay"] < 48 * 60]

# some info
print("Num facilities: {}".format(len(parkings["SiteID"].unique())))
print("Num parking events: {}".format(len(parkings)))

min_dt = "2019-06-03"
max_dt = "2019-06-09"

parkings = parkings[
    (parkings["EntryDateTime"] >= min_dt) & (parkings["EntryDateTime"] <= max_dt)
]


def sample_electric_vehicles(df, ev_share, seed=42):
    """
    Selects whether parked vehicle is an EV with probability p=ev_share. Sets energy demand of vehicles
    that have been set to ICE to 0 kWh
    :param df:
    :param ev_share: [0,1]
    :param seed: seed of random generator, if undefined set to None
    :return: adds colums EV_yn and "final_kWhRequested_updated"
    """
    # NOTE: TESTED AND PASSED

    if ev_share > 1 or ev_share < 0:
        raise ValueError("EV share must be in interval [0,1]!!!")

    # instantiate random number generator
    rng = default_rng(seed=seed)

    # randomly select EV with p=ev_share
    ev_array = rng.choice([0, 1], p=[1 - ev_share, ev_share], size=len(df))

    # match to df
    df["EV_yn"] = ev_array

    # update energy demand, set to 0 if not an EV
    # df["final_kWhRequested_updated"] = df["EV_yn"] * df["final_kWhRequested"]

    return df


parkings = sample_electric_vehicles(parkings, ev_share=1, seed=42)


### Request Charging Y/N

# (for now just have a min threshold)


def sample_charge_request_yn(df, min_stay_minutes):
    """
    xxx
    :param df:
    :param ev_share: [0,1]
    :param seed: seed of random generator, if undefined set to None
    :return: adds colums EV_yn and "final_kWhRequested_updated"
    """

    # Conditional Expectation of Success given Planned Duration of Stay
    # Use probability in Brenoulli Process to define whether there is a charge request

    # For now: assume minimum stay duration of XX

    df["request_charge_yn"] = df.apply(
        lambda x: 1 if x.EV_yn == 1 and x.MinutesStay >= min_stay_minutes else 0, axis=1
    )

    return df


### Charge Quantity Request

#### Collaborative Filtering Appraoch
# (DO THIS AFTER DAYS HAVE BEEN SELECTED DUE TO LONG RUNTIME)


def generate_energy_demand_coll_filter_approach(
    df, ev_share, min_stay_minutes, max_charge_rate, seed
):
    df = sample_electric_vehicles(df, ev_share, seed=seed)
    df = sample_charge_request_yn(df, min_stay_minutes)

    # update charge request

    df["final_kWhRequested_Coll_Filter"] = df.apply(
        lambda x: (
            x.userInputs_kWhRequested
            if x.request_charge_yn == 1 and x.EV_yn == 1
            else 0
        ),
        axis=1,
    )

    # ensure realistic charge request (i.e., fulfillment physically possible at max charge rate)
    df["final_kWhRequested_Coll_Filter"] = df.apply(
        lambda x: (
            x.MinutesStay / 60 * max_charge_rate
            if x.final_kWhRequested_Coll_Filter >= x.MinutesStay / 60 * max_charge_rate
            else x.final_kWhRequested_Coll_Filter
        ),
        axis=1,
    )

    # ensure that energy demand does not exceed typical max battery size (70kWh)
    max_battery_size = 100  # kWh
    df["final_kWhRequested_Coll_Filter"] = df["final_kWhRequested_Coll_Filter"].apply(
        lambda x: max_battery_size if x > max_battery_size else x
    )

    df["final_kWhRequested_Coll_Filter"] = df["final_kWhRequested_Coll_Filter"].apply(
        lambda x: round(x, ndigits=2)
    )

    return df
parkings['EntryDateTime'] = pd.to_datetime(parkings['EntryDateTime'])
parkings['time_bucket'] = parkings['EntryDateTime'].dt.floor('2H').dt.hour

parkings = generate_energy_demand_coll_filter_approach(
    df=parkings, ev_share=1, min_stay_minutes=20, max_charge_rate=50, seed=42
)


def get_laxity(df):
    df["park_duration_h"] = df["MinutesStay"] / 60
    df["full_speed_charge_duration"] = df["final_kWhRequested_Coll_Filter"] / 22
    df["laxity"] = df["park_duration_h"] - df["full_speed_charge_duration"]

    return df


parkings = get_laxity(parkings)
parkings = parkings[
    (parkings["SiteID"] == "Facility_3")
    & (parkings["final_kWhRequested_Coll_Filter"] > 0)
]
def remove_outliers(df,columns):
    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1

        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        df = df[
            (df[column] >= lower_bound) & (df[column] <= upper_bound)
            ]
    return df

parkings = remove_outliers(parkings, ['park_duration_h', 'full_speed_charge_duration'])

parkings = parkings[(parkings["EntryDate"] >= "2019-06-03") & (parkings["EntryDate"] <= "2019-06-11")]
palette = sns.color_palette("mako", n_colors=1)

parkings['arrival_dis'] = 1
parking = parkings.groupby('time_bucket').sum('arrival_dis').reset_index()
# parking['arrival_dis'] = parkings['arrival_dis']
print(parking)


# parkings = parkings.melt(id_vars='time_bucket', value_vars=['park_duration_h', 'arrival_dis'],
#                     var_name='Group', value_name='Values')


fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(12, 4), constrained_layout=True)
sns.boxplot(data=parkings, y="park_duration_h", x="time_bucket", palette = palette, ax=ax[2])
sns.boxplot(data=parkings, y="final_kWhRequested_Coll_Filter", x="time_bucket", palette = palette, ax=ax[1])
set_plotting_style()
colors = sns.color_palette('mako', n_colors=1)
color_arrival = colors[0]
# color_departure = colors[-2]
sns.distplot(parkings["time_bucket"], bins=12,
            hist=True,
            kde=False,
            norm_hist=True,
            ax=ax[0],
            label="Arrivals",
            color=color_arrival)

ax[0].set_xlabel('Hour', fontsize=12)
# ax[0].set_ylim(0.2, 0.9)
ax[0].set_ylabel('Probability', fontsize=12)
ax[0].set_title(f'Distribution of Vehicle Arrival Time', fontsize=12)
handles, labels = plt.gca().get_legend_handles_labels()

ax[1].set_xlabel('Hour', fontsize=12)
# ax[0].set_ylim(0.2, 0.9)
ax[1].set_ylabel('Energy [kWh]', fontsize=12)
ax[1].set_title(f'Maximum Energy Demand of EV Users', fontsize=12)
handles, labels = plt.gca().get_legend_handles_labels()
# new_labels = ['low', 'medium', 'high', 'very high']
# ax[0].legend(handles, new_labels)

ax[2].set_xlabel('Hour', fontsize=12)
# ax[1].set_legend(loc='lower right', fontsize=12)
ax[2].set_ylabel('Duration [h]', fontsize=12)
# ax[1].set_ylim(0.2, 0.9)
ax[2].set_title(f'Park Duration of EV Users', fontsize=12)
handles, labels = plt.gca().get_legend_handles_labels()
# new_labels = ['low', 'medium', 'high', 'very high']
# ax[1].legend(handles, new_labels)

plt.tight_layout()
plt.savefig("User_preferences.pdf")
plt.show()

# fig, ax = plt.subplots(figsize=(10, 5))
# order_list = [
#     "Business",
#     "Morning_short",
#     "Afternoon_short",
#     "Evening_short",
#     "Overnight",
#     "Long-term",
# ]
# sns.boxplot(y="laxity", x="ClusterName", data=parkings, ax=ax, order=order_list)
# ax.set_ylabel("Laxity [h]", fontsize=16)
# ax.set_xlabel("Cluster", fontsize=16)
# ax.set_xticklabels(
#     [
#         "Business",
#         "Morning\n (short)",
#         "Afternoon\n (short)",
#         "Evening\n (short)",
#         "Overnight",
#         "Long-term",
#     ]
# )
# plt.tight_layout()
# plt.savefig("laxity_per_parker_type.pdf")
# plt.show()

# data_path = "/home/rahadi/Projects/EVCC/kshroer/EV_Charging_Clusters/Data/EV_Energy_Demand_Data/"
#
# travel_demand = pd.read_csv(data_path+"CA_Travel_survey_trip.csv")
# travel_demand = travel_demand[["sampno","perno","tripno","trpmiles","distance_mi","trvlcmin","trptrans17"]]
#
#
# travel_demand_cars = travel_demand[travel_demand["trptrans17"].isin([3,4,6,5,18])==True]
#
# travel_demand_cars_grouped = travel_demand_cars.groupby("sampno").sum()
# # remove unreasonably large numbers
# travel_demand_cars_grouped = travel_demand_cars_grouped[travel_demand_cars_grouped["distance_mi"]<=travel_demand_cars_grouped["distance_mi"].quantile(q=0.99)]
# # remove unreasonably small numbers (likely roundtrips)
# travel_demand_cars_grouped = travel_demand_cars_grouped[travel_demand_cars_grouped["distance_mi"]>0.2]
#
# sns.distplot(travel_demand_cars_grouped["distance_mi"])
#
#
# def generate_daily_distance_traveled(df, travel_demand_path=data_path + "CA_Travel_survey_trip.csv", location="US",
#                                      seed=42):
#     us_mean_km = 25.9 * 1.60934  # https://tedb.ornl.gov/data/ Table 9.09
#     de_mean_km = (
#                              1051 + 1149) / 2 / 30  # https://mobilitaetspanel.ifv.kit.edu/english/downloads.php; 2019 vehicle miles
#
#     # assume exponential distribution
#     # It can be shown for the exponential distribution that the mean is equal to the standard deviation; i.e., μ = σ = 1/λ
#
#     rng = default_rng(seed=seed)
#
#     if location == "US":
#
#         # get data and prepare
#         travel_demand = pd.read_csv(travel_demand_path)
#         travel_demand = travel_demand[
#             ["sampno", "perno", "tripno", "trpmiles", "distance_mi", "trvlcmin", "trptrans17"]]
#         # limit to relevant transport modes only
#         travel_demand_cars = travel_demand[travel_demand["trptrans17"].isin([3, 4, 6, 5, 18]) == True]
#         # group by sample and process
#         travel_demand_cars_grouped = travel_demand_cars.groupby("sampno").sum()
#         # remove unreasonably large numbers
#         travel_demand_cars_grouped = travel_demand_cars_grouped[
#             travel_demand_cars_grouped["distance_mi"] <= travel_demand_cars_grouped["distance_mi"].quantile(q=0.99)]
#         # remove unreasonably small numbers (likely roundtrips)
#         travel_demand_cars_grouped = travel_demand_cars_grouped[travel_demand_cars_grouped["distance_mi"] > 0.2]
#
#         # sample randomly with repalcement (need to fit an empirical distribution here!)
#         distances = travel_demand_cars_grouped.sample(n=len(df), replace=True, random_state=seed)
#         distances = distances["distance_mi"] * 1.60934
#         distances = np.array(distances)
#
#         # distances = rng.exponential(us_mean_km, size=len(df))
#
#     elif location == "DE":
#         distances = rng.exponential(de_mean_km, size=len(df))
#
#     df["daily_travel_km"] = distances
#
#     return df
#
#
# def generate_home_charging_opportunity(df, home_charging_share, seed=42):
#     # Brenoulli Process that assigns home charging with probability p(HomeCharging) = home_charging_share
#
#     rng = default_rng(seed=seed)
#
#     home_charging_array = rng.choice([0, 1], p=[1 - home_charging_share, home_charging_share], size=len(df))
#
#     df["home_charging_yn"] = home_charging_array
#
#     return df
#
#
# def generate_energy_demand_travel_based_approach(df, ev_share=0.5, min_stay_minutes=20, home_charging_share=0.8,
#                                                  max_charge_rate=22,
#                                                  location="US", seed=42):
#     df = sample_electric_vehicles(df, ev_share=ev_share, seed=seed)
#     df = sample_charge_request_yn(df, min_stay_minutes=min_stay_minutes)
#     df = generate_daily_distance_traveled(df, location=location, seed=seed)
#     df = generate_home_charging_opportunity(df, home_charging_share=home_charging_share, seed=seed)
#
#     # energy per km
#     kWh_per_km = 18.5 / 100  # Tesal Model S75
#     kWh_per_km = round(0.2306631886256587, 3)  # mean of ACN sample
#
#     # base charging demand'
#     df["final_kWhRequested_travel_demand"] = df.apply(
#         lambda x: x.daily_travel_km * kWh_per_km if x.request_charge_yn == 1 and x.EV_yn == 1 else 0, axis=1)
#
#     # charging energy (if home charging = yes, divide by 2)
#     # df["final_kWhRequested_travel_demand"] = df.apply(lambda x: x.final_kWhRequested_travel_demand/2 if x.home_charging_yn==1 else x.final_kWhRequested_travel_demand, axis=1)
#
#     # ensure realistic charge request (i.e., fulfillment physically possible at max charge rate)
#     df["final_kWhRequested_travel_demand"] = df.apply(lambda
#                                                           x: x.MinutesStay / 60 * max_charge_rate if x.final_kWhRequested_travel_demand >= x.MinutesStay / 60 * max_charge_rate
#     else x.final_kWhRequested_travel_demand, axis=1)
#
#     max_battery_size = 100  # kWh
#     df["final_kWhRequested_travel_demand"] = df["final_kWhRequested_travel_demand"].apply(
#         lambda x: max_battery_size if x > max_battery_size else x)
#
#     # round
#     df["final_kWhRequested_travel_demand"] = df["final_kWhRequested_travel_demand"].apply(lambda x: round(x, ndigits=2))
#
#     return df
#
#
# parkings = generate_energy_demand_travel_based_approach(df=parkings,ev_share=0.25,min_stay_minutes=20,home_charging_share=0.8,max_charge_rate=22,location="US",seed=42)
#
# fig,ax = plt.subplots(figsize=(16,9))
#
# sns.distplot(parkings[parkings["request_charge_yn"]==1]["final_kWhRequested_Coll_Filter"],ax=ax)
# sns.distplot(parkings[parkings["request_charge_yn"]==1]["final_kWhRequested_travel_demand"],ax=ax,color="red")

# sns.distplot(parkings["final_kWhRequested_Coll_Filter"],ax=ax)
# sns.distplot(parkings["final_kWhRequested_travel_demand"],ax=ax,color="red")
