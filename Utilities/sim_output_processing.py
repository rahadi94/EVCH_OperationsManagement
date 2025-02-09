# Process sim data
import random

import pandas as pd
from datetime import timedelta

from Environment.helper.configuration.configuration import Configuration


def get_requests(raw_output_save_path, post_fix=""):
    """
    Some basic processing of request output data
    :return:
    """

    file_name = "requests{}.csv".format(post_fix)
    df = pd.read_csv(raw_output_save_path + file_name)

    df["arrival_time"] = pd.to_datetime(df["arrival_time"])
    df["departure_time"] = pd.to_datetime(df["departure_time"])
    df["day"] = df["arrival_time"].apply(lambda x: x.date())
    df["day_of_week"] = df["arrival_time"].apply(lambda x: x.weekday())
    df["weekday_yn"] = df["day_of_week"].apply(lambda x: 1 if x < 5 else 0)
    df["park_duration_min"] = df["departure_period"] - df["arrival_period"]

    # hours
    df["arrival_hour"] = df["arrival_time"].apply(lambda dt: dt.hour)
    df["departure_hour"] = df["departure_time"].apply(lambda dt: dt.hour)

    # minutes
    # df["minutes_from_midnight_arrival"] = df["datetime"].apply(lambda x: minutes_from_midnight(x))
    # df["minutes_from_midnight_base5"] = df["minutes_from_midnight"].apply(lambda x: round_down_to_base(x, 5))

    # laxity
    df["park_duration_h"] = df["park_duration_min"] / 60
    df["full_speed_charge_duration"] = df["energy_requested"] / 22
    df["laxity"] = df["park_duration_h"] - df["full_speed_charge_duration"]

    df["service_level"] = df["energy_charged"] / df["energy_requested"]
    df["service_level"] = df.apply(
        lambda x: 1 if x.energy_requested == 0 else x.service_level, axis=1
    )
    df["service_level"] = df["service_level"].apply(lambda x: 1 if x > 1 else x)
    df["service_level"] = df["service_level"].apply(lambda x: 0 if x < 0 else x)

    return df


def get_storage_load(raw_output_save_path, post_fix="", sim_start_date="2019-06-03"):
    """
    Some basic processing of request output data
    :return:
    """

    file_name = "storage{}.csv".format(post_fix)
    df = pd.read_csv(raw_output_save_path + file_name)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    df.reset_index(inplace=True)
    df.rename({"index": "sim_time"}, inplace=True, axis=1)
    df["sim_time"] = df["sim_time"].apply(lambda x: int(x))

    # add actual datetime
    df["datetime"] = df["sim_time"].apply(
        lambda x: pd.to_datetime(sim_start_date) + timedelta(minutes=x)
    )
    df["day"] = df["datetime"].apply(lambda x: x.date())

    df["day_of_week"] = df["datetime"].apply(lambda x: x.weekday())
    dow_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    df["day_of_week_string"] = df["day_of_week"].apply(lambda x: dow_dict[x])

    df["weekday_yn"] = df["day_of_week"].apply(lambda x: 1 if x < 5 else 0)
    wd_dict = {0: "Weekend", 1: "Weekday"}
    df["weekday_yn_string"] = df["weekday_yn"].apply(lambda x: wd_dict[x])

    df["hour"] = df["datetime"].apply(lambda x: x.hour)
    df["minutes_from_midnight"] = df["datetime"].apply(
        lambda x: minutes_from_midnight(x)
    )
    df["minutes_from_midnight_base5"] = df["minutes_from_midnight"].apply(
        lambda x: round_down_to_base(x, 5)
    )

    return df


def get_historical_prices(
    raw_output_save_path, post_fix="", sim_start_date="2019-06-03"
):
    """
    Some basic processing of request output data
    :return:
    """

    file_name = "price_history{}.csv".format(post_fix)
    df = pd.read_csv(raw_output_save_path + file_name)
    df.drop(columns=["Unnamed: 0"], inplace=True)
    # df.columns = Configuration.instance().power

    df.reset_index(inplace=True)
    df.rename({"index": "sim_time"}, inplace=True, axis=1)
    df["sim_time"] = df["sim_time"].apply(lambda x: int(x))

    df["datetime"] = df["sim_time"].apply(
        lambda x: pd.to_datetime(sim_start_date) + timedelta(minutes=x)
    )
    df["day"] = df["datetime"].apply(lambda x: x.date())

    df["day_of_week"] = df["datetime"].apply(lambda x: x.weekday())
    dow_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    df["day_of_week_string"] = df["day_of_week"].apply(lambda x: dow_dict[x])

    df["weekday_yn"] = df["day_of_week"].apply(lambda x: 1 if x < 5 else 0)
    wd_dict = {0: "Weekend", 1: "Weekday"}
    df["weekday_yn_string"] = df["weekday_yn"].apply(lambda x: wd_dict[x])

    df["hour"] = df["datetime"].apply(lambda x: x.hour)
    df["minutes_from_midnight"] = df["datetime"].apply(
        lambda x: minutes_from_midnight(x)
    )
    df["minutes_from_midnight_base5"] = df["minutes_from_midnight"].apply(
        lambda x: round_down_to_base(x, 5)
    )

    return df


def get_episode_rewards(raw_output_save_path, post_fix="", sim_start_date="2019-06-03"):
    df = pd.DataFrame()
    for i in [
        "Dynamic-Cap-Pricing",
        "Time-of-Use",
        "Perfect-Info",
        "Dynamic-Menu-Based",
        "Dynamic-Traditional",
    ]:
        if i == "Dynamic-Cap-Pricing":
            file_name = f"training_results_pricing_double_PV_p0_alpha_low.csv"
            inner_df = pd.read_csv(raw_output_save_path + file_name)
            inner_df.drop(columns=["Unnamed: 0"], inplace=True)
            df[i] = inner_df["profit"].iloc[0:1000]
        elif i == "Dynamic-Menu-Based":
            file_name = f"training_results_pricing_double_menu_1000_50_average_power_m_200_h.csv"
            inner_df = pd.read_csv(raw_output_save_path + file_name)
            inner_df.drop(columns=["Unnamed: 0"], inplace=True)
            df[i] = inner_df["profit"].iloc[0:1000]
        elif i == "Dynamic-Traditional":
            file_name = f"training_results_pricing_double_PV_p0_low.csv"
            inner_df = pd.read_csv(raw_output_save_path + file_name)
            inner_df.drop(columns=["Unnamed: 0"], inplace=True)
            df[i] = inner_df["profit"].iloc[0:1000]
        elif i == "Time-of-Use":
            df[i] = 381
        elif i == "Perfect-Info":
            df[i] = 4270
        elif i == "Baseline":
            df[i] = 2320
        df[i] = df[i].rolling(window=10).mean()

    df.fillna(method="ffill", inplace=True)
    df = df.iloc[::20]
    return df


def get_occupancy(requests_df, agg_level=5):
    """
    calculates occupancy from requests
    :param requests_df:
    :param agg_level:
    :return:
    """

    # settings
    agg_level = agg_level  # minutes

    # set entry/exit time to target res
    requests_df["arrival_time"] = requests_df["arrival_time"].apply(
        lambda x: x.replace(second=0, minute=round_down_to_base(x.minute, agg_level))
    )
    requests_df["departure_time"] = requests_df["departure_time"].apply(
        lambda x: x.replace(second=0, minute=round_down_to_base(x.minute, agg_level))
    )  # .apply(lambda x: x.replace(second=0, microsecond=0))

    # count entries and exits per time period in separate df
    entry_counts = requests_df.groupby(
        ["facility", "ev_yn", "user_type", "arrival_time"], as_index=False
    ).agg({"vehicle_id": "count"})
    entry_counts.columns = ["facility", "ev_yn", "user_type", "time", "entry_counts"]
    entry_counts["key"] = entry_counts.apply(
        lambda x: "{}_{}_{}_{}".format(x.facility, x.ev_yn, x.user_type, x.time), axis=1
    )

    exit_counts = requests_df.groupby(
        ["facility", "ev_yn", "user_type", "departure_time"], as_index=False
    ).agg({"vehicle_id": "count"})
    exit_counts.columns = ["facility", "ev_yn", "user_type", "time", "exit_counts"]
    exit_counts["key"] = exit_counts.apply(
        lambda x: "{}_{}_{}_{}".format(x.facility, x.ev_yn, x.user_type, x.time), axis=1
    )

    # compute target index
    target_index = pd.DataFrame()
    requests_df["arrival_time"] = requests_df["arrival_time"].apply(
        lambda x: x.replace(second=0)
    )
    requests_df["departure_time"] = requests_df["arrival_time"].apply(
        lambda x: x.replace(second=0)
    )
    for facility in requests_df["facility"].unique():
        for ev in requests_df["ev_yn"].unique():
            for cluster in requests_df["user_type"].unique():
                out = pd.DataFrame()
                out["time"] = pd.date_range(
                    start=requests_df["arrival_time"].min(),
                    end=requests_df["departure_time"].max(),
                    freq="{}S".format(agg_level * 60),
                )
                out["facility"] = facility
                out["ev_yn"] = ev
                out["user_type"] = cluster
                out["key"] = out.apply(
                    lambda x: "{}_{}_{}_{}".format(
                        x.facility, x.ev_yn, x.user_type, x.time
                    ),
                    axis=1,
                )

                # print(facility,cluster)
                target_index = target_index.append(out)

    # merge into occupancy df
    occupancy_df = target_index.merge(
        entry_counts[["key", "entry_counts"]], on="key", how="left"
    )
    occupancy_df = occupancy_df.merge(
        exit_counts[["key", "exit_counts"]], on="key", how="left"
    )
    occupancy_df.fillna(0, inplace=True)  # fill periods with no entries/exits
    occupancy_df["net_occupancy_change"] = (
        occupancy_df["entry_counts"] - occupancy_df["exit_counts"]
    )  # get net occupancy change

    # compute occupancy per facility, user type, ev
    occupancy_final = pd.DataFrame()

    for facility in occupancy_df["facility"].unique():
        for ev in occupancy_df["ev_yn"].unique():
            for cluster in occupancy_df["user_type"].unique():
                out_df = occupancy_df[
                    (occupancy_df["facility"] == facility)
                    & (occupancy_df["ev_yn"] == ev)
                    & (occupancy_df["user_type"] == cluster)
                ]

                out_df.sort_values(by="time", inplace=True, ascending=True)
                out_df["total_occupancy"] = out_df["net_occupancy_change"].cumsum()

                occupancy_final = occupancy_final.append(out_df)

    # times
    occupancy_final["date"] = occupancy_final["time"].apply(lambda x: x.date())
    occupancy_final["hour"] = occupancy_final["time"].apply(lambda x: x.hour)
    occupancy_final["minutes_from_midnight"] = occupancy_final["time"].apply(
        lambda x: minutes_from_midnight(x)
    )

    return occupancy_final


def get_load_curve(
    raw_output_save_path,
    post_fix,
    sim_start_date="2019-06-03",
    agent_name=Configuration.instance().pricing_agent_name,
):
    """
    Retrieves CS utilization data and compute load curve
    :param raw_output_save_path:
    :param SIM_START_DAY:
    :return:
    """

    # get data
    file_name = f"CSs{post_fix}_{agent_name}.csv"
    df = pd.read_csv(raw_output_save_path + file_name)
    df.drop("Unnamed: 0", inplace=True, axis=1)

    # filter to consumption
    df_out = df[df["info"] == "kWh_consumption"]
    df_out.drop(["cs_id", "info"], inplace=True, axis=1)
    df_out.reset_index(inplace=True, drop=True)

    # transpose (to get time as index)
    df_out = df_out.transpose()
    df_out.reset_index(inplace=True)
    df_out.rename({"index": "sim_time"}, inplace=True, axis=1)
    df_out["total_consumption"] = df_out.sum(axis=1)
    df_out["total_load"] = df_out["total_consumption"] * 60
    df_out["sim_time"] = df_out["sim_time"].apply(lambda x: int(x))

    # add actual datetime
    df_out["datetime"] = df_out["sim_time"].apply(
        lambda x: pd.to_datetime(sim_start_date) + timedelta(minutes=x)
    )
    df_out["day"] = df_out["datetime"].apply(lambda x: x.date())

    df_out["day_of_week"] = df_out["datetime"].apply(lambda x: x.weekday())
    dow_dict = {
        0: "Monday",
        1: "Tuesday",
        2: "Wednesday",
        3: "Thursday",
        4: "Friday",
        5: "Saturday",
        6: "Sunday",
    }
    df_out["day_of_week_string"] = df_out["day_of_week"].apply(lambda x: dow_dict[x])

    df_out["weekday_yn"] = df_out["day_of_week"].apply(lambda x: 1 if x < 5 else 0)
    wd_dict = {0: "Weekend", 1: "Weekday"}
    df_out["weekday_yn_string"] = df_out["weekday_yn"].apply(lambda x: wd_dict[x])

    df_out["hour"] = df_out["datetime"].apply(lambda x: x.hour)
    df_out["minutes_from_midnight"] = df_out["datetime"].apply(
        lambda x: minutes_from_midnight(x)
    )
    df_out["minutes_from_midnight_base5"] = df_out["minutes_from_midnight"].apply(
        lambda x: round_down_to_base(x, 5)
    )

    return df_out


def get_CS_utilization(raw_output_save_path, post_fix):
    """

    :param raw_output_save_path: location of saved output
    :return:
    """
    # get data
    file_name = "CSs{}.csv".format(post_fix)
    df = pd.read_csv(raw_output_save_path + file_name)
    df.drop("Unnamed: 0", inplace=True, axis=1)

    # "num_vehicles_connected","num_vehicles_charging","kWh_consumption"

    # get charging info
    df_charging_status = df[df["info"] == "num_vehicles_charging"]
    df_charging_status.drop(["cs_id", "info"], axis=1, inplace=True)
    df_charging_status.reset_index(inplace=True, drop=True)

    total_periods = len(df_charging_status.columns)
    total_days = total_periods / 60 / 24
    df_charging_status["total_vehicle_charging_periods"] = df_charging_status.sum(
        axis=1
    )
    df_charging_status["total_periods"] = total_periods
    df_charging_status["total_days"] = total_days

    # get connection info
    df_connection_status = df[df["info"] == "num_vehicles_connected"]
    df_connection_status.drop(["cs_id", "info"], axis=1, inplace=True)
    df_connection_status.reset_index(inplace=True, drop=True)
    df_connection_status["total_vehicle_connection_periods"] = df_connection_status.sum(
        axis=1
    )

    # get consumption info
    df_energy_status = df[df["info"] == "kWh_consumption"]
    df_energy_status.drop(["cs_id", "info"], axis=1, inplace=True)
    df_energy_status.reset_index(inplace=True, drop=True)
    df_energy_status["total_energy_delivered"] = df_energy_status.sum(axis=1)

    # merge and combine
    df_out = df_charging_status.join(
        pd.DataFrame(df_connection_status["total_vehicle_connection_periods"])
    )
    df_out = df_out.join(pd.DataFrame(df_energy_status["total_energy_delivered"]))
    df_out = df_out[
        [
            "total_vehicle_charging_periods",
            "total_periods",
            "total_days",
            "total_vehicle_connection_periods",
            "total_energy_delivered",
        ]
    ]

    df_out.reset_index(inplace=True)
    df_out.rename({"index": "cs_id"}, axis=1, inplace=True)

    # add additional util indicators
    df_out["max_vehicle_periods"] = df_out["total_periods"] * 6
    df_out["max_energy_delivered"] = df_out["total_periods"] * 22 / 60
    df_out["utilization_energy_transfer"] = (
        df_out["total_energy_delivered"] / df_out["max_energy_delivered"]
    )
    df_out["daily_energy_transfer"] = (
        df_out["total_energy_delivered"] / df_out["total_days"]
    )

    return df_charging_status, df_connection_status, df_energy_status, df_out


##########################################
# HELPERS
def round_down_to_base(num, base):
    return num - (num % base)


def minutes_from_midnight(dt):
    mfm = dt.hour * 60 + dt.minute
    return mfm
