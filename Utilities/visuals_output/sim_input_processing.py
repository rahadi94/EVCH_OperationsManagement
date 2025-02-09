# required libraries
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta
import pytz
import numpy.random
from numpy.random import default_rng
import sklearn
from sklearn.neighbors import KernelDensity


#######################################################################################
### COMBINED ROUTINES


def get_sim_charging_requests(
    base_path,
    facility_list,
    ev_share,
    geography,
    sim_start_day="2019-06-03",
    sim_duration=1,
    candidate_days=False,
    n_days=False,
    day_types=["Workday", "Saturday", "Sunday"],
    facility_size_limit=False,
):
    """
    Combined processing and sampling routine
    :param base_path: path containing data sub-folders
    :param facility_list: list of facilities to be included
    :param ev_share: share of EVs in total parking events [0,1]
    :param geography: geography on which empirical distribution of vehicle models is based (for sampling of battery sizes);
                      select from "DE", "CA", "TX", "NY", "FL", "CO"
    :param sim_start_day: start day of simulation (default: '2019-06-03')
    :param sim_duration: duration in days (int)
    :param day_types: types of days to include (default = ["Workday","Saturday","Sunday"])
    :param facility_size_limit: int of facility size, if no limit ==False
    :return:
    """

    sim_start_day = pd.to_datetime(sim_start_day)
    # days_list = list(pd.date_range(start=sim_start_day, periods=sim_duration, freq="D"))

    ### Process and sample raw preference data and pre-process
    preferences = get_raw_charging_preference_data(base_path=base_path)

    ### Select subset of facilities in scope (can be just 1)
    preferences = select_facilities(df=preferences, facility_names=facility_list)

    #### limit to selected day types
    preferences = select_day_types(df=preferences, day_types=day_types)
    preferences = preferences[
        preferences["EntryDate"] >= sim_start_day
    ]  # limit to days after start date!

    #### get day list
    days_list = preferences["EntryDate"].unique()
    days_list.sort()
    days_list = list(days_list[0:sim_duration])
    print("Simulation will be run for the following days:", days_list)

    #### Select subset of days in scope and compute sim_times
    if sim_start_day and sim_duration:
        preferences = preferences[preferences["EntryDate"].isin(days_list) == True]
        preferences = compute_sim_times(df=preferences, sim_start_day=sim_start_day)
        print(
            "Simulation will be run for a total of {} parking events:".format(
                len(preferences)
            )
        )
    elif n_days and candidate_days:  # TODO: Not currently propoerly implemented!
        preferences = sample_candidate_days(
            df=preferences, n_days=n_days, candidate_days=candidate_days, seed=42
        )

    ### Sample daily events such that max_occupancy is met per each facility (OPTIONAL IF SIZE OF FACILITY IS TO BE CONTROLLED)
    if facility_size_limit:
        preferences = sample_daily_requests_up_to_capacity(
            df=preferences,
            max_occupancy=facility_size_limit,
            margin=0.005,
            replace=False,
            seed=42,
        )
    ### sample EVs based on share
    preferences = sample_electric_vehicles(df=preferences, ev_share=ev_share, seed=42)

    ### sample battery size per each EV
    preferences = get_battery_size(
        df=preferences, base_path=base_path, geography=geography
    )

    preferences.sort_values(by="EntryDateTime", inplace=True)
    preferences.reset_index(drop=True, inplace=True)

    # days_list = list(preferences["EntryDate"].unique())

    return preferences  # , days_list


def get_sim_baseload_curve(
    base_path,
    facility="K1+K2",
    sim_start_day="2019-06-03",
    sim_duration=1,
    min_facility_baseload=100,
    max_facility_baseload=700,
):
    """

    :param base_path: data file
    :param facility: available facilities: ['K1', 'K2', 'K1+K2, 'H3', 'M1', 'M1p', 'M2', 'B2a', 'B1', 'B2b', 'H1', 'H2','H4', 'H6']
    :param sim_start_day: start of simulation
    :param sim_duration: duration (in days) of simulation
    :param min_facility_baseload: minimum facility baseload
    :param max_facility_baseload: maximum facility baseload
    :return: returns scale-adjusted (to selected min/max baseload) baseload curve over entire duration of simulation (in 1-minute resolution)
    """

    # get avg scaled load curve
    scaled_load = get_scaled_baseload_data(base_path=base_path, facility=facility)

    # get index
    target_index_df = get_load_data_target_index(
        sim_start_day=sim_start_day, sim_duration=sim_duration
    )

    # get scaled load curve for sim duration
    out_df = target_index_df.merge(
        scaled_load, on=["month", "daytype", "time"], how="left"
    )

    # fill missing minutes
    out_df.fillna(method="ffill", inplace=True)

    # rescale to target scale
    out_df["load_kw_rescaled"] = out_df["load_kw_scaled"].apply(
        lambda x: (x * (max_facility_baseload - min_facility_baseload))
        + min_facility_baseload
    )

    out_df = out_df[["sim_time", "time", "load_kw_rescaled"]]
    out_df.set_index("sim_time", inplace=True)

    return out_df


def get_sim_PV_load_factors(
    base_path,
    sim_start_day,
    sim_duration,
    time_col="time",
    load_factor_col="DE_pv_national_current",
    pv_data_resolution="hour",
):
    """
    Produces dataframe with (hourly) PV load factors per simulation period.
    Actual load can be readily computed by multiplying with size of panels
    :param base_path:
    :param sim_start_day:
    :param sim_duration:
    :param time_col:
    :param load_factor_col:
    :param pv_data_resolution:
    :return:
    """

    # data source: https://www.nrel.gov/grid/solar-power-data.html
    # data source: https://data.open-power-system-data.org/ninja_pv_wind_profiles/

    file = "PV_Data/pv_profiles_EU.csv"  # update this at later stage. Ideally want 5min output!
    sim_start_day = pd.to_datetime(sim_start_day)

    pv_load_factors_df = pd.read_csv(base_path + file)
    pv_load_factors_df[time_col] = pd.to_datetime(pv_load_factors_df[time_col])
    pv_load_factors_df[time_col] = pv_load_factors_df[time_col].apply(
        lambda ts: ts.replace(tzinfo=None)
    )
    pv_load_factors_df = pv_load_factors_df[[time_col, load_factor_col]]

    index_df = get_load_data_target_index(
        sim_start_day=sim_start_day, sim_duration=sim_duration
    )

    # merge onto target index
    if pv_data_resolution == "hour":
        # create hourly index
        index_df["timestamp"] = pd.to_datetime(index_df["timestamp"])
        index_df[time_col] = index_df["timestamp"].apply(
            lambda ts: ts.replace(minute=0, second=0)
        )
        index_df[time_col] = index_df[time_col].apply(
            lambda ts: ts.replace(tzinfo=None)
        )
        # merge
        out = index_df.merge(pv_load_factors_df, how="left", on=time_col)

    # tidy up cols
    out.rename({load_factor_col: "pv_load_factor"}, axis=1, inplace=True)
    cols = ["sim_time", time_col, "pv_load_factor"]
    out = out[cols]
    out.set_index("sim_time", inplace=True)
    return out


#######################################################################################
### FUNCTIONS


def get_raw_charging_preference_data(
    base_path, sim_start_day="2019-06-03", inter_arrival_time=False
):
    """
    Load and return pre-processed preference data as DataFrame object
    :param base_path: path to folder containing all sub-folders of preference data
    :return: DataFrame object with raw data
    NOTE: TESTED AND PASSED
    """

    sim_start_day = pd.to_datetime(sim_start_day)

    file_loc = "EV_Energy_Demand_Data/Parking+Charging_Data_BLENDED_CLUSTERED.csv"
    df = pd.read_csv(os.path.join(base_path, file_loc))

    # match facility_type
    facility_dict = {
        "Facility_1": "Mixed-use_Facility",
        "Facility_2": "Hospital",
        "Facility_3": "Mixed-use_Facility",
        "Facility_4": "Destination_Facility",
        "Facility_5": "Workplace_Facility",
        "Facility_6": "Workplace_Facility",
        "Facility_KoeBogen": "Destination_Facility",
    }

    df["FacilityType"] = df["SiteID"].apply(lambda facility: facility_dict[facility])

    # ensure datetime format
    df["EntryDateTime"] = pd.to_datetime(df["EntryDateTime"])
    df["ExitDateTime"] = pd.to_datetime(df["ExitDateTime"])
    df["EntryDate"] = pd.to_datetime(df["EntryDate"])
    df["ExitDate"] = pd.to_datetime(df["ExitDate"])

    # Day Type Flag
    df["DayType"] = df["EntryDayOfWeek"].apply(lambda x: day_classifier(x))

    # Inter Arrival Time
    if inter_arrival_time:
        facilities = df["SiteID"].unique()
        combined_df = pd.DataFrame()
        for facility in facilities:
            df_fac = df[df["SiteID"] == facility]

            df_fac = df_fac.sort_values(by="EntryDateTime", ascending=True)
            df_fac["EntryDateTimeShifted"] = df_fac.EntryDateTime.shift(1)
            df_fac["TimeSinceLastEntry"] = (
                df_fac["EntryDateTime"] - df_fac["EntryDateTimeShifted"]
            )
            df_fac["TimeSinceLastEntry"] = df_fac["TimeSinceLastEntry"].fillna(
                method="bfill"
            )

            combined_df = combined_df.append(df_fac)

        combined_df["MinutesSinceLastArrival"] = combined_df[
            "TimeSinceLastEntry"
        ].apply(
            lambda td: time_delta_to_scalar_minutes(td)
        )  # TODO FIX
        combined_df["SecondsSinceLastArrival"] = (
            combined_df["MinutesSinceLastArrival"] * 60
        )

        # limit to required columns
        cols = [
            "EntryDateTime",
            "ExitDateTime",
            "HoursStay",
            "MinutesStay",
            "RevenueAmount",
            "SiteID",
            "Year",
            "MinutesSinceLastArrival",
            "SecondsSinceLastArrival",
            "EntryDate",
            "DayType",
            "ExitDate",
            "EntryHour",
            "ExitHour",
            "EntryDayOfWeek",
            "EntryWeekday_yn",
            "EntryHoliday_yn",
            "userInputs_WhPerMile",
            "userInputs_kWhRequested",
            "userInputs_milesRequested",
            "MaxFeasible_kwhRequested",
            "final_kWhRequested",
            "ClusterNum",
            "ClusterName",
            "FacilityType",
        ]

        combined_df = combined_df[cols]

        return combined_df

    else:

        cols = [
            "EntryDateTime",
            "ExitDateTime",
            "HoursStay",
            "MinutesStay",
            "RevenueAmount",
            "SiteID",
            "Year",
            "EntryDate",
            "DayType",
            "ExitDate",
            "EntryHour",
            "ExitHour",
            "EntryDayOfWeek",
            "EntryWeekday_yn",
            "EntryHoliday_yn",
            "userInputs_WhPerMile",
            "userInputs_kWhRequested",
            "userInputs_milesRequested",
            "MaxFeasible_kwhRequested",
            "final_kWhRequested",
            "ClusterNum",
            "ClusterName",
            "FacilityType",
        ]

        df = df[cols]

        return df


def get_scaled_baseload_data(base_path, facility="K1+K2"):
    """
    Gets avg minute-level scaled baseload curve per month, daytype (workday, saturday, sunday)
    :param base_path:
    :param facility: available facilities: ['K1', 'K2', 'K1+K2, 'H3', 'M1', 'M1p', 'M2', 'B2a', 'B1', 'B2b', 'H1', 'H2','H4', 'H6']
    :return:
    """

    file = "Building_Load_Profile_Data/Lastgang.csv"
    dateparser = lambda x: datetime.strptime(x, "%d/%m/%Y %H:%M")

    loads_df = pd.read_csv(
        base_path + file,
        parse_dates=True,
        index_col="Time",
        sep=",",
        date_parser=dateparser,
    )
    loads_df["datetime"] = loads_df.index
    loads_df.reset_index(inplace=True)
    loads_df["K1+K2"] = loads_df["K1"] + loads_df["K2"]

    # print(loads_df.head())

    # limit to facility in scope
    loads_df = loads_df[["datetime", facility]]

    # Adjust Building Loads which are provided in kWh (K1,K2,B1,B2,H2,H3,H4,H6)
    if facility in ["K1", "K2", "K1+K2", "B2a", "B1", "B2b", "H2", "H3", "H4", "H6"]:
        # for col in ["K1","K2","K1+K2",'B2a', 'B1', 'B2b',"H2","H3","H4","H6"]:
        loads_df[facility] = (
            loads_df[facility] * 4
        )  # approximation of peak load, 1 period is 15minutes, i.e. quarter hour

    # add additional temporal features
    loads_df["datetime_hour"] = loads_df["datetime"].apply(
        lambda x: x.replace(minute=0, second=0)
    )
    loads_df["time"] = loads_df["datetime"].apply(lambda x: x.time())

    loads_df["hour"] = loads_df["datetime"].apply(lambda x: x.hour)
    loads_df["dayofweek"] = loads_df["datetime"].apply(lambda x: x.weekday())
    loads_df["weeknum"] = loads_df["datetime"].apply(lambda x: x.isocalendar()[1])
    loads_df["isweekday"] = loads_df["dayofweek"].apply(lambda x: 1 if x < 5 else 0)
    loads_df["daytype"] = loads_df["dayofweek"].apply(lambda x: day_classifier(x))
    loads_df["month"] = loads_df["datetime"].apply(lambda x: x.month)
    loads_df["year"] = loads_df["datetime"].apply(lambda x: x.year)
    loads_df.reset_index(drop=True, inplace=True)

    # limit to 2019, for which we have parking data
    loads_df = loads_df[loads_df["year"].isin([2018, 2019])]

    # get avg load curve per each month and day type!
    # cols=['K1', 'K2', 'H3', 'M1', 'M1p', 'M2', 'B2a', 'B1', 'B2b', 'H1', 'H2','H4', 'H6', 'K1+K2','timestamp','isweekday', 'month']
    cols = [facility, "time", "daytype", "month"]
    avg_load = (
        loads_df[cols].groupby(["month", "daytype", "time"], as_index=False).mean()
    )
    avg_load.dropna(axis=1, inplace=True)
    avg_load.columns = ["month", "daytype", "time", "load_kw"]

    avg_load_scaled = min_max_scaling(df=avg_load, cols_to_scale=["load_kw"])

    return avg_load_scaled


def get_load_data_target_index(sim_start_day="2019-06-03", sim_duration=1):
    """
    Computes simulation target index for baseload data to be matched on
    :param sim_start_day:
    :param sim_duration:
    :return:
    """
    # generate index
    target_index_df = list(
        pd.date_range(
            start=pd.to_datetime(sim_start_day),
            periods=(sim_duration * 24 * 60),
            freq="min",
        )
    )
    target_index_df = pd.DataFrame(target_index_df)
    target_index_df.reset_index(inplace=True)
    target_index_df.columns = ["sim_time", "timestamp"]
    target_index_df["date"] = target_index_df["timestamp"].apply(lambda ts: ts.date())
    target_index_df["time"] = target_index_df["timestamp"].apply(lambda ts: ts.time())

    # get required time proporeties
    target_index_df["dayofweek"] = target_index_df["date"].apply(lambda x: x.weekday())
    target_index_df["isweekday"] = target_index_df["dayofweek"].apply(
        lambda x: 1 if x < 5 else 0
    )
    target_index_df["daytype"] = target_index_df["dayofweek"].apply(
        lambda x: day_classifier(x)
    )
    target_index_df["month"] = target_index_df["date"].apply(lambda x: x.month)

    return target_index_df


def compute_sim_times(df, sim_start_day):
    """
    Function computes sim times relative to sim_start_day (in minutes)
    :param df:
    :param sim_start_day:
    :return:
    """

    sim_start_day = pd.to_datetime(sim_start_day)

    # Compute Minutes from Sim Start
    df["StayDuration"] = df["ExitDateTime"] - df["EntryDateTime"]
    df["StayDurationMinutes"] = df["StayDuration"].apply(
        lambda td: time_delta_to_scalar_minutes(td)
    )

    # get timedelta
    df["EntryMinutesFromSimStart"] = df["EntryDateTime"].apply(
        lambda x: x - sim_start_day
    )
    df["ExitMinutesFromSimStart"] = df["ExitDateTime"].apply(
        lambda x: x - sim_start_day
    )

    # get scalar minutes
    df["EntryMinutesFromSimStart"] = df["EntryMinutesFromSimStart"].apply(
        lambda x: time_delta_to_scalar_minutes(x)
    )
    df["ExitMinutesFromSimStart"] = df["ExitMinutesFromSimStart"].apply(
        lambda x: time_delta_to_scalar_minutes(x)
    )

    return df


def sample_candidate_days(df, n_days, candidate_days, seed=42):
    """
    Sample n=n_days from list of candidate days (with or without replacement) and return df with this subsample
    :param df:
    :param n_days: number of samples to be drawn from candidates, if replace=False this cannot exceed the number of candidates!
    :param candidate_days: list of candidate days in datetime format
    :param replace: whether drawn samples are returned and can be drawn again
    :param seed: seed of random generator, if undefined set to None
    :return:
    """
    # NOTE: TESTED AND PASSED

    if len(candidate_days) < n_days:
        raise ValueError(
            "Number of days to be sampled cannot exceed number of candidate days!!!"
        )

    # ensure candidate_days are in dt format
    candidate_days_dt = []
    for day in candidate_days:
        dt = pd.to_datetime(day)
        candidate_days_dt.append(dt)

    # instantiate random number generator and choose days randomly from candidates
    rng = default_rng(seed=seed)
    days = rng.choice(
        candidate_days_dt, size=n_days, replace=False
    )  # we do not want to sample the same day twice, therefore set replace=False

    df = df[df["EntryDate"].isin(days) == True]

    return df


def select_facilities(df, facility_names):
    """
    Selects subset of preference data that belongs to facilities in facility_names list
    :param df: preference data as DataFrame
    :param facility_names: list of facilities
    :return: DataFrame containing events that belong to facilities in facility_names
    """
    # NOTE: TESTED AND PASSED

    # ensure this is an iterable list
    facility_names = list(facility_names)

    # select subset of events belonging to facilities
    df = df[(df["SiteID"].isin(facility_names) == True)]

    return df


def select_day_types(df, day_types):
    """
    Limits df to date types
    :param df:
    :param day_types: list of day types. Select from ["Workday","Saturday","Sunday"]
    :return: masked df
    """

    df = df[(df["DayType"].isin(day_types) == True)]

    return df


def sample_daily_requests_up_to_capacity(
    df, max_occupancy, temp_res=5, margin=0.005, replace=False, seed=42, verbose=False
):
    """
    Function samples sufficient parking events up to the desired capacity for each facility and day contained in df
    :param df: base df of parking events
    :param facility_names: list of facilities (can only be 1)
    :param max_occupancy: desired max. occupancy
    :param temp_res: temporal resolution at which occupancy is computed (in minutes)
    :param margin: margin (in %) on deviation from max occupancy
    :param replace: sample with replacement?
    :param seed: fix see or not for replication purposes
    :return:
    """

    if margin > 1 or margin < 0:
        raise ValueError("Margin (%) must be in interval [0,1]!!!")

    # set entry/exit time to target res
    temp_res = temp_res  # minutes!
    df["EntryDateTime{}min".format(temp_res)] = df["EntryDateTime"].apply(
        lambda x: x.replace(second=0, minute=round_down_to_base(x.minute, temp_res))
    )
    df["ExitDateTime{}min".format(temp_res)] = df["ExitDateTime"].apply(
        lambda x: x.replace(second=0, minute=round_down_to_base(x.minute, temp_res))
    )

    # per each facility and day sample events up to target occupation and collect

    final_df = pd.DataFrame(columns=df.columns)
    for facility_name in df["SiteID"].unique():

        out_df = pd.DataFrame(columns=df.columns)
        for day in df["EntryDate"].unique():

            # select facility and day
            day_df = df[(df["SiteID"] == facility_name) & (df["EntryDate"] == day)]

            # sample events up to max_occupancy
            event_count = round(
                max_occupancy * 2.5
            )  # initialize to 2.5 the max occupancy (seems like a good value)
            while True:
                # sample n=event_count events
                day_df_sampled = day_df.sample(
                    n=event_count, replace=replace, random_state=seed
                )

                # simulate occupancy over day and find max
                ## count entries and exits per time period in separate df
                entry_counts = (
                    day_df_sampled[
                        [
                            "EntryDateTime{}min".format(temp_res),
                            "ExitDateTime{}min".format(temp_res),
                        ]
                    ]
                    .groupby(["EntryDateTime{}min".format(temp_res)])
                    .count()
                )
                entry_counts.reset_index(inplace=True)
                entry_counts.columns = ["DateTime{}min".format(temp_res), "EntryCounts"]

                exit_counts = (
                    day_df_sampled[
                        [
                            "EntryDateTime{}min".format(temp_res),
                            "ExitDateTime{}min".format(temp_res),
                        ]
                    ]
                    .groupby(["ExitDateTime{}min".format(temp_res)])
                    .count()
                )
                exit_counts.reset_index(inplace=True)
                exit_counts.columns = ["DateTime{}min".format(temp_res), "ExitCounts"]

                ## compute target index
                target_index = pd.DataFrame()
                target_index["DateTime{}min".format(temp_res)] = pd.date_range(
                    start=day_df_sampled["EntryDateTime{}min".format(temp_res)].min(),
                    end=day_df_sampled["EntryDateTime{}min".format(temp_res)].max(),
                    freq="{}S".format(temp_res * 60),
                )
                ## merge entry and exit counts and compute total occupancy per interval
                occupancy = target_index.merge(
                    entry_counts, on="DateTime{}min".format(temp_res), how="left"
                )
                occupancy = occupancy.merge(
                    exit_counts, on="DateTime{}min".format(temp_res), how="left"
                )
                occupancy.fillna(0, inplace=True)
                occupancy["NetOccupancyChange"] = (
                    occupancy["EntryCounts"] - occupancy["ExitCounts"]
                )
                occupancy["TotalOccupancy"] = occupancy[
                    "NetOccupancyChange"
                ].cumsum()  # gets cumulative sum of occupancy change!

                max_daily_occ = max(occupancy["TotalOccupancy"])
                if verbose:
                    print(max_daily_occ)

                # update event count if not max_occupancy not within margin, else break
                if max_daily_occ > round(max_occupancy + (margin * max_occupancy)):
                    event_count += (
                        -1
                    )  # reduce event counts if above max_occupancy + margin
                    if verbose:
                        print("occupancy too large")
                elif max_daily_occ < round(max_occupancy - (margin * max_occupancy)):
                    event_count += 1  # increase event counts if below max_occupancy
                    if verbose:
                        print("occupancy too small")
                elif np.absolute(max_daily_occ - max_occupancy) <= round(
                    margin * max_occupancy
                ):
                    print(
                        "Facility {}; Day {}: occupancy within margin at {} parking events".format(
                            facility_name, day, event_count
                        )
                    )
                    break  # break loop if within margin

            # append to out_df
            out_df = out_df.append(day_df_sampled)

        # append facility df to final df
        final_df = final_df.append(out_df)

    return final_df


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
    df["final_kWhRequested_updated"] = df["EV_yn"] * df["final_kWhRequested"]

    return df


def get_battery_size(df, base_path, geography="CA"):
    """
    Samples battery size based on empirical distribution of EV stock in selected geography
    :param df:
    :param geography: select from DE, CA, TX, NY, FL, CO --> This is for choosing which empirical distribution of vehicle stock to use
    :return: Column "BatterySize"
    """

    file_loc = "EV_Stock_Data/EV_Stock_Data.xlsx"

    # get batteries
    batteries = pd.read_excel(os.path.join(base_path, file_loc), engine="openpyxl")  #
    batteries = batteries[
        [
            "Model",
            "Type",
            "Battery Capacity Combined (in kWh)",
            "Consumption Upper (kWh/100km)",
            "{}_Units".format(geography),
        ]
    ]

    # limit to passenger cars and drop NaNs
    batteries = batteries[batteries["Type"] == "passenger car"]
    batteries.dropna(inplace=True)

    # get vehicle_model probability
    batteries["Vehicle_probability"] = batteries["{}_Units".format(geography)].apply(
        lambda x: x / (batteries["{}_Units".format(geography)].sum())
    )

    # sample battery capa based on empirical distribution of vehicle stock in chosen geography
    rng = default_rng(seed=42)
    capacity = batteries["Battery Capacity Combined (in kWh)"]
    probability = batteries["Vehicle_probability"]

    df["BatterySize"] = df["EV_yn"].apply(
        lambda x: (
            rng.choice(capacity, p=probability, replace=False) if x == 1 else np.NaN
        )
    )

    return df


# This is a three-step process using empirical distributions
# Good introductory resource is here: https://machinelearningmastery.com/empirical-distribution-function-in-python/


############ ALTERNATIVE APPROACH: Sample each vehicle independently (prior or during simulation)


def get_empirical_dist_IAT(raw_pref_data):
    """
    Calculates distibutions of inter-arrival time (IAT) conditional on facility, day class ["Workday","Saturday","Sunday"] and hour
    :param raw_pref_data: empirical data
    :return: kde_dict_IAT with key [facility][day][hour] --> fitted kde for that facility, day and hour
    """

    # prepare data
    facilities = raw_pref_data["SiteID"].unique()

    combined_df = pd.DataFrame()
    for facility in facilities:
        df = raw_pref_data[raw_pref_data["SiteID"] == facility]

        df.sort_values(by="EntryDateTime", ascending=True, inplace=True)
        df["Entry_shifted"] = df.EntryDateTime.shift(1)
        df["TimeSinceLastEntry"] = df["EntryDateTime"] - df["Entry_shifted"]
        df["TimeSinceLastEntry"] = df["TimeSinceLastEntry"].fillna(method="bfill")

        combined_df = combined_df.append(df)

    combined_df["MinutesSinceLastArrivals"] = combined_df["TimeSinceLastEntry"].apply(
        lambda td: time_delta_to_scalar_minutes(td)
    )
    combined_df["SecondsSinceLastArrivals"] = (
        combined_df["MinutesSinceLastArrivals"] * 60
    )
    combined_df["DayType"] = combined_df["EntryDayOfWeek"].apply(
        lambda x: day_classifier(x)
    )

    # compute non-parameteric densities using KDEs conditional on facility, day, hour

    bandwidth = 7  # TODO: Implement corss-validation to select optimal bandwidth
    kde_dict_IAT = dict()
    for facility in facilities:

        day_dict = dict()
        for day in ["Workday", "Saturday", "Sunday"]:

            hour_dict = dict()
            for hour in range(0, 24, 1):

                df = combined_df[
                    (combined_df["SiteID"] == facility)
                    & (
                        combined_df["SecondsBetweenArrivals"]
                        <= combined_df["SecondsBetweenArrivals"].quantile(q=0.998)
                    )  # only select 99.8% quantile
                    & (combined_df["EntryHour"] == hour)
                    & (combined_df["DayType"] == day)
                ]

                if len(df) > 0:

                    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
                    kde.fit(np.array(df["SecondsBetweenArrivals"]).reshape(-1, 1))

                else:
                    kde = "no samples"

                hour_dict[hour] = kde

            day_dict[day] = hour_dict

        kde_dict_IAT[facility] = day_dict

    return kde_dict_IAT


def get_empirical_dist_AR(raw_pref_data):
    """
    Calculates distributions of arrival rate (AR) conditional on facility, day class ["Workday","Saturday","Sunday"] and hour
    :param data: raw_pref_data
    :return: kde_dict_AR with key [facility][day][hour] --> fitted kde of arrival rates for that facility, day and hour
    """

    # prepare data
    facilities = raw_pref_data["SiteID"].unique()

    # get entries per minute
    raw_pref_data["EntryDateHour"] = raw_pref_data["EntryDateTime"].apply(
        lambda x: x.replace(minute=0, second=0)
    )
    parkings_per_hour = (
        raw_pref_data[["EntryDateHour", "EntryDateTime", "SiteID"]]
        .groupby(["EntryDateHour", "SiteID"])
        .count()
    )
    parkings_per_hour.columns = ["NumEntries"]
    parkings_per_hour.reset_index()

    # get target df
    target_df_final = pd.DataFrame()

    target_df = pd.DataFrame(
        index=pd.date_range(
            start=raw_pref_data["EntryDateTime"].min().replace(minute=0, second=0),
            end=raw_pref_data["EntryDateTime"].max().replace(minute=0, second=0),
            freq="H",
        )
    )

    for facility in ["Facility_KoeBogen"]:
        df = target_df
        df["SiteID"] = facility
        df.reset_index(inplace=True)
        df.columns = ["EntryDateHour", "SiteID"]

        target_df_final = target_df_final.append(df)

    # merge entry counts
    AR_df = target_df_final.merge(
        parkings_per_hour, how="left", on=["EntryDateHour", "SiteID"]
    )
    AR_df.fillna(0, inplace=True)
    AR_df.reset_index(inplace=True, drop=True)
    AR_df["VehiclesPerMinute"] = AR_df["NumEntries"] / 60

    # add additional meta data
    AR_df["EntryHour"] = AR_df["EntryDateHour"].apply(lambda x: x.hour)
    AR_df["EntryDayOfWeek"] = AR_df["EntryDateHour"].apply(lambda x: x.weekday())
    AR_df["EntryWeekday_yn"] = AR_df["EntryDayOfWeek"].apply(
        lambda x: 1 if x < 5 else 0
    )
    AR_df["DayType"] = AR_df["EntryDayOfWeek"].apply(lambda x: day_classifier(x))

    # compute non-parameteric densities using KDEs conditional on facility, day, hour

    bandwidth = 7  # TODO: Implement cross-validation to select optimal bandwidth
    kde_dict_AR = dict()
    for facility in facilities:

        day_dict = dict()
        for day in ["Workday", "Saturday", "Sunday"]:

            hour_dict = dict()
            for hour in range(0, 24, 1):

                df = AR_df[
                    (AR_df["SiteID"] == facility)
                    & (
                        AR_df["VehiclesPerMinute"]
                        <= AR_df["VehiclesPerMinute"].quantile(q=0.998)
                    )
                    & (AR_df["EntryHour"] == hour)
                    & (AR_df["DayType"] == day)
                ]

                if len(df) > 0:

                    kde = KernelDensity(bandwidth=bandwidth, kernel="gaussian")
                    kde.fit(np.array(df["VehiclesPerMinute"]).reshape(-1, 1))

                else:
                    kde = "no samples"

                hour_dict[hour] = kde

            day_dict[day] = hour_dict

        kde_dict_AR[facility] = day_dict

    return kde_dict_AR


#######################################################################################
### HELPER FUNCTIONS


def time_delta_to_scalar_minutes(td):
    """
    Converts time delta to scalar minutes
    :param td: time delta
    :return: total_minutes (scalar)
    """
    full_minutes, seconds = divmod(td.seconds, 60)

    total_minutes = td.days * 24 * 60 + full_minutes + seconds / 60

    return total_minutes


def day_classifier(day_of_week):
    """
    Classifies days into ["Workday","Saturday","Sunday"]
    :param day_of_week: day of week [0,6]
    :return: day type (string)
    """

    if day_of_week < 5:
        out = "Workday"
    elif day_of_week == 5:
        out = "Saturday"
    else:
        out = "Sunday"

    return out


def min_max_scaling(df, cols_to_scale):
    """
    Completes a min-max scaling for selected columns
    :param df:
    :param cols_to_scale: list of columns to scale
    :return:
    """

    for col in cols_to_scale:
        col_max = df[col].max()
        col_min = df[col].min()
        col_range = col_max - col_min

        df["{}_scaled".format(col)] = df[col].apply(lambda x: (x - col_min) / col_range)

    return df


def adjust_time(x):
    """
    helped function to fix datetime format
    :param x:
    :return:
    """

    return x.strftime("%H:%M:%S")


############# LEGACY FUNCTION ###############
