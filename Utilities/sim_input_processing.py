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
from Environment.log import lg


def sample_training_and_test_weeks(year=2019, seed=42):
    """
    Randomly samples one week per month
    :param sim_season:
    :param summer_start:
    :param summer_end:
    :param year:
    :return:
    """

    year_days = pd.DataFrame()
    year_days["Day"] = pd.date_range(
        start=pd.to_datetime("{}-01-01".format(year)),
        end=pd.to_datetime("{}-12-31".format(year)),
        freq="D",
    )
    year_days["IsMonday"] = year_days["Day"].apply(
        lambda x: 1 if x.weekday() == 0 else 0
    )
    year_days["Month"] = year_days["Day"].apply(lambda x: x.month)

    relevant_weeks = year_days[(year_days["IsMonday"] == 1)]

    # sample training weeks (1 per month)
    training_weeks = []
    for month in relevant_weeks["Month"].unique():
        sample = relevant_weeks[(relevant_weeks["Month"] == month)]
        sampled_week = sample.sample(1, random_state=seed)["Day"].iloc[0]

        training_weeks.append(sampled_week)

    # remove sampled training weeks from relevant_weeks
    relevant_weeks = relevant_weeks[relevant_weeks["Day"].isin(training_weeks) == False]

    test_weeks = []
    for month in relevant_weeks["Month"].unique():
        sample = relevant_weeks[(relevant_weeks["Month"] == month)]
        sampled_week = sample.sample(1, random_state=seed)["Day"].iloc[0]

        test_weeks.append(sampled_week)

    return training_weeks, test_weeks


def sample_week(sim_seasons, summer_start, summer_end, year=2019, seed=None):
    """
    Samples a week from the given season
    :param sim_season:
    :param summer_start:
    :param summer_end:
    :param year:
    :return:
    """
    summer_start = pd.to_datetime(summer_start)
    summer_end = pd.to_datetime(summer_end)
    # compute season datetimes
    # year_days = pd.DataFrame(pd.DatetimeIndex(start="2019-01-01".format(year), end="2019-12-31".format(year), freq="D"),
    #                         columns=["Day"])
    year_days = pd.DataFrame()
    year_days["Day"] = pd.date_range(
        start=pd.to_datetime("{}-01-01".format(year)),
        end=pd.to_datetime("{}-12-31".format(year)),
        freq="D",
    )
    year_days["IsMonday"] = year_days["Day"].apply(
        lambda x: 1 if x.weekday() == 0 else 0
    )
    year_days["Season"] = year_days["Day"].apply(
        lambda x: "Summer" if x >= summer_start and x <= summer_end else "Winter"
    )

    relevant_weeks = year_days[
        (year_days["IsMonday"] == 1) & (year_days["Season"].isin(sim_seasons))
    ]

    if seed:
        sampled_week = relevant_weeks.sample(1, random_state=seed)["Day"].iloc[0]
        last_day_week = sampled_week + timedelta(days=6)

        if (
            last_day_week.year > year
        ):  # probably do not need this if statement here but just keep for good measure
            i = 1
            while last_day_week.year > year:
                sampled_week = relevant_weeks.sample(1, random_state=seed + i)[
                    "Day"
                ].iloc[0]
                last_day_week = sampled_week + timedelta(days=6)
                i += 1
    else:
        sampled_week = relevant_weeks.sample(1)["Day"].iloc[0]
        last_day_week = sampled_week + timedelta(days=6)

        while last_day_week.year > year:
            sampled_week = relevant_weeks.sample(1)["Day"].iloc[0]
            last_day_week = sampled_week + timedelta(days=6)

    return sampled_week

    # get list of all Mondays in season


def get_sim_charging_requests(
    base_path,
    cache_path,
    facility_list,
    ev_share,
    geography,
    max_charge_rate,
    sim_start_day,
    parking_capacity,
    demand_gen_approach="charging_demand_data",
    sim_duration=1,
    sim_seasons=["Summer"],
    summer_start_date=pd.to_datetime("2019-04-29"),
    summer_end_date=pd.to_datetime("2019-10-06"),
    candidate_days=False,
    n_days=False,
    day_types=["Workday", "Saturday", "Sunday"],
    limit_requests_to_capa=False,
):
    """
    Combined processing and sampling routine
    :param cache_path:
    :param sim_season: for which season to simulate
    :param summer_end_date:
    :param summer_start_date:
    :param base_path: path containing data sub-folders
    :param facility_list: list of facilities to be included
    :param ev_share: share of EVs in total parking events [0,1]
    :param geography: geography on which empirical distribution of vehicle models is based (for sampling of battery sizes);
                      select from "DE", "CA", "TX", "NY", "FL", "CO"
    :param max_charge_rate:
    :param demand_gen_approach: approach based on which charging demand is calculated []
    :param sim_start_day: start day of simulation (default: '2019-06-03')
    :param sim_duration: duration in days (int)
    :param candidate_days:
    :param n_days:
    :param day_types: days to include: ["Workday","Saturday","Sunday"]
    :param limit_requests_to_capa: boolean of whether to limit capacity or not, True if yes, False if no
    :return:
    """

    sim_start_day = pd.to_datetime(sim_start_day)

    if (
        sim_start_day and sim_duration and not limit_requests_to_capa
    ):  # if some start date and sim duration is specified execute below routine
        try:  # load data if in cache
            print(
                cache_path
                + "preferences_{}_{}_{}_{}_{}.pkl".format(
                    sim_seasons, facility_list, day_types, ev_share, max_charge_rate
                )
            )
            if type(max_charge_rate) == int:
                parking_charging_preferences = pd.read_pickle(
                    cache_path
                    + "preferences_{}_{}_{}_{}_{}.pkl".format(
                        sim_seasons, facility_list, day_types, ev_share, max_charge_rate
                    )
                )
            elif type(max_charge_rate) == dict:
                parking_charging_preferences = pd.read_pickle(
                    cache_path
                    + "preferences_{}_{}_{}_{}_{}.pkl".format(
                        sim_seasons,
                        facility_list,
                        day_types,
                        ev_share,
                        max_charge_rate["fast"],
                    )
                )
        except (FileNotFoundError, PermissionError):  # compute if not in cache
            lg.warning("Sample must be computed")
            ### Process and sample raw preference data and pre-process
            parking_preferences = get_raw_parking_data(base_path=base_path)
            ### Select subset of facilities in scope (can be just 1)
            parking_preferences = select_facilities(
                df=parking_preferences, facility_names=facility_list
            )
            #### limit to selected day types
            parking_preferences = select_day_types(
                df=parking_preferences, day_types=day_types
            )
            #### limit to season
            parking_preferences["Season"] = parking_preferences["EntryDate"].apply(
                lambda x: (
                    "Summer"
                    if x >= summer_start_date and x <= summer_end_date
                    else "Winter"
                )
            )
            parking_preferences = parking_preferences[
                parking_preferences["Season"].isin(sim_seasons)
            ]
            parking_preferences.drop(columns=["Season"], inplace=True)

            ### Get Charging Demand
            parking_charging_preferences = generate_charging_demand_data(
                parking_data=parking_preferences,
                base_path=base_path,
                demand_gen_approach=demand_gen_approach,
                geography=geography,
                ev_share=ev_share,
                min_stay_minutes=20,
                home_charging_share=0.8,
                max_charge_rate=max_charge_rate,
                seed=42,
            )
            ## save (full year data) to cache for faster access in future runs
            if type(max_charge_rate) == int:
                parking_charging_preferences.to_pickle(
                    cache_path
                    + "preferences_{}_{}_{}_{}_{}.pkl".format(
                        sim_seasons, facility_list, day_types, ev_share, max_charge_rate
                    )
                )
            elif type(max_charge_rate) == dict:
                parking_charging_preferences.to_pickle(
                    cache_path
                    + "preferences_{}_{}_{}_{}_{}.pkl".format(
                        sim_seasons,
                        facility_list,
                        day_types,
                        ev_share,
                        max_charge_rate["fast"],
                    )
                )

        #### limit to selected days
        parking_charging_preferences = parking_charging_preferences[
            parking_charging_preferences["EntryDate"] >= sim_start_day
        ]  # limit to days after start date!
        days_list = parking_charging_preferences["EntryDate"].unique()
        days_list.sort()
        days_list = list(days_list[0:sim_duration])
        parking_charging_preferences = parking_charging_preferences[
            parking_charging_preferences["EntryDate"].isin(days_list) == True
        ]  # ensure
        # parking_preferences = parking_preferences[parking_preferences["ExitDate"].isin(days_list) == True] # remove entries that stay beyond end of sim

        # compute sim times
        parking_charging_preferences = compute_sim_times(
            df=parking_charging_preferences, sim_start_day=sim_start_day
        )
        lg.warning(
            "Simulation will be run for the following days:",
            parking_charging_preferences["EntryDate"].unique(),
        )

    elif sim_start_day and sim_duration and limit_requests_to_capa:
        try:  # load data if pre-prepared!
            df_out = pd.DataFrame()
            for fac in facility_list:
                if type(max_charge_rate) == int:
                    parking_charging_preferences = pd.read_pickle(
                        cache_path
                        + "requests_sample_{}_size_limit_{}_max_charge_rate_{}.pkl".format(
                            fac, parking_capacity, max_charge_rate
                        )
                    )
                elif type(max_charge_rate) == dict:
                    parking_charging_preferences = pd.read_pickle(
                        cache_path
                        + "requests_sample_{}_size_limit_{}_max_charge_rate_{}.pkl".format(
                            fac, parking_capacity, max_charge_rate["fast"]
                        )
                    )
                df_out = df_out.append(parking_charging_preferences)
        except (FileNotFoundError, PermissionError):  # compute if not in cache
            lg.warning("Sample must be computed")
            df_out = pd.DataFrame()
            for fac in facility_list:
                ### Process and sample raw preference data and pre-process
                parking_preferences = get_raw_parking_data(base_path=base_path)
                ### Select subset of facilities in scope (can be just 1)
                parking_preferences = select_facilities(
                    df=parking_preferences, facility_names=[fac]
                )
                #### limit to selected day types
                parking_preferences = select_day_types(
                    df=parking_preferences, day_types=day_types
                )
                #### limit to season
                parking_preferences["Season"] = parking_preferences["EntryDate"].apply(
                    lambda x: (
                        "Summer"
                        if x >= summer_start_date and x <= summer_end_date
                        else "Winter"
                    )
                )
                parking_preferences = parking_preferences[
                    parking_preferences["Season"].isin(sim_seasons)
                ]
                parking_preferences.drop(columns=["Season"], inplace=True)

                ### Get Charging Demand
                parking_charging_preferences = generate_charging_demand_data(
                    parking_data=parking_preferences,
                    base_path=base_path,
                    demand_gen_approach=demand_gen_approach,
                    geography=geography,
                    ev_share=1,
                    min_stay_minutes=20,
                    home_charging_share=0.8,
                    max_charge_rate=max_charge_rate,
                    seed=42,
                )

                ### Sample daily events such that max_occupancy is met per each facility (OPTIONAL IF SIZE OF FACILITY IS TO BE CONTROLLED)
                print("CHECKPOINT reached")
                parking_charging_preferences = (
                    sample_up_to_capacity_inc_proportionality_seasonality(
                        df=parking_charging_preferences,
                        facility_capa=parking_capacity,
                        margin=0.005,
                        seed=42,
                    )
                )
                df_out = df_out.append(parking_charging_preferences)
            # save to pickle
            if type(max_charge_rate) == int:
                df_out.to_pickle(
                    cache_path
                    + "requests_sample_{}_size_limit_{}_max_charge_rate_{}.pkl".format(
                        fac, parking_capacity, max_charge_rate
                    )
                )
            elif type(max_charge_rate) == dict:
                df_out.to_pickle(
                    cache_path
                    + "requests_sample_{}_size_limit_{}_max_charge_rate_{}.pkl".format(
                        fac, parking_capacity, max_charge_rate["fast"]
                    )
                )

        #### select EVs based on ev_share
        parking_charging_preferences = df_out
        parking_charging_preferences = generate_electric_vehicles_yn(
            df=parking_charging_preferences, ev_share=ev_share, seed=42
        )
        parking_charging_preferences["final_kWhRequested_updated"] = (
            parking_charging_preferences["final_kWhRequested_updated"]
            * parking_charging_preferences["EV_yn"]
        )

        #### limit to selected days
        parking_charging_preferences["EntryDate"] = pd.to_datetime(
            parking_charging_preferences["EntryDate"]
        ).apply(lambda x: x.date())
        parking_charging_preferences = parking_charging_preferences[
            parking_charging_preferences["EntryDate"] >= sim_start_day
        ]  # limit to days after start date!
        days_list = parking_charging_preferences["EntryDate"].unique()
        days_list.sort()
        days_list = list(days_list[0:sim_duration])
        parking_charging_preferences = parking_charging_preferences[
            parking_charging_preferences["EntryDate"].isin(days_list) == True
        ]  # ensure
        # parking_preferences = parking_preferences[parking_preferences["ExitDate"].isin(days_list) == True] # remove entries that stay beyond end of sim
        lg.warning(parking_charging_preferences.head(2))
        # compute sim times
        parking_charging_preferences = compute_sim_times(
            df=parking_charging_preferences, sim_start_day=sim_start_day
        )

        parking_charging_preferences.reset_index(inplace=True, drop=True)

        lg.warning(
            "Simulation will be run for the following days:",
            parking_charging_preferences["EntryDate"].unique(),
        )

    elif n_days and candidate_days and not limit_requests_to_capa:

        raise ValueError("Not Currently Implemented ")

    return parking_charging_preferences


def get_sim_baseload_curve(
    base_path,
    cache_path,
    facility="K1+K2",
    sim_start_day="2019-06-03",
    num_lookback_periods=24 * 60,
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
    scaled_load = get_scaled_baseload_data(
        base_path=base_path, cache_path=cache_path, facility=facility
    )

    # get index
    # target_index_df = get_load_data_target_index(sim_start_day=sim_start_day, sim_duration=sim_duration)
    target_index_df = get_load_data_target_index(
        sim_start_day=sim_start_day,
        num_lookback_periods=num_lookback_periods,
        sim_duration=sim_duration,
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
    cache_path,
    sim_start_day="2019-06-03",
    num_lookback_periods=24 * 60,
    sim_duration=1,
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

    try:
        pv_load_factors_df = pd.read_pickle(cache_path + "pv_loads.pkl")
    except (FileNotFoundError, PermissionError):
        file = "PV_Data/pv_profiles_EU.csv"  # update this at later stage. Ideally want 5min output!
        sim_start_day = pd.to_datetime(sim_start_day)

        pv_load_factors_df = pd.read_csv(base_path + file)
        pv_load_factors_df[time_col] = pd.to_datetime(pv_load_factors_df[time_col])
        pv_load_factors_df[time_col] = pv_load_factors_df[time_col].apply(
            lambda ts: ts.replace(tzinfo=None)
        )
        pv_load_factors_df = pv_load_factors_df[[time_col, load_factor_col]]
        pv_load_factors_df.to_pickle(cache_path + "pv_loads.pkl")

    # index_df = get_load_data_target_index(sim_start_day=sim_start_day,sim_duration=sim_duration)
    index_df = get_load_data_target_index(
        sim_start_day=sim_start_day,
        num_lookback_periods=num_lookback_periods,
        sim_duration=sim_duration,
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
    out["hour"] = out["time"].apply(lambda x: x.hour)
    out["day"] = out["time"].apply(lambda x: x.day)
    return out


#######################################################################################
### HELPER FUNCTIONS


def get_raw_parking_data(
    base_path, sim_start_day="2019-06-03", inter_arrival_time=False
):
    """
    Load and return pre-processed preference data as DataFrame object
    :param base_path: path to folder containing all sub-folders of preference data
    :return: DataFrame object with raw data
    NOTE: TESTED AND PASSED
    """

    sim_start_day = pd.to_datetime(sim_start_day)

    file_loc = "EV_Energy_Demand_Data/Parking+Charging_Data_BLENDED_CLUSTERED_v2.csv"
    df = pd.read_csv(os.path.join(base_path, file_loc))

    # limit parking duration to 48h (let's not do this for now!)
    df = df[df["MinutesStay"] < 48 * 60]

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
    df["EntryDate"] = df["EntryDateTime"].apply(lambda x: x.date())
    df["ExitDate"] = df["ExitDateTime"].apply(lambda x: x.date())

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
            "userInputs_kWhPerkm",
            "userInputs_kWhRequested",
            "userInputs_kmRequested",
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
            "userInputs_kWhPerkm",
            "userInputs_kWhRequested",
            "userInputs_kmRequested",
            "MaxFeasible_kwhRequested",
            "final_kWhRequested",
            "ClusterNum",
            "ClusterName",
            "FacilityType",
        ]

        df = df[cols]

        return df


def get_scaled_baseload_data(base_path, cache_path, facility="K1+K2"):
    """
    Gets avg minute-level scaled baseload curve per month, daytype (workday, saturday, sunday)
    :param base_path:
    :param facility: available facilities: ['K1', 'K2', 'K1+K2, 'H3', 'M1', 'M1p', 'M2', 'B2a', 'B1', 'B2b', 'H1', 'H2','H4', 'H6']
    :return:
    """
    try:
        avg_load_scaled = pd.read_pickle(cache_path + "scaled_baseload.pkl")
    except (FileNotFoundError, PermissionError):
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
        if facility in [
            "K1",
            "K2",
            "K1+K2",
            "B2a",
            "B1",
            "B2b",
            "H2",
            "H3",
            "H4",
            "H6",
        ]:
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
        avg_load_scaled.to_pickle(cache_path + "scaled_baseload.pkl")

    return avg_load_scaled


def get_load_data_target_index(
    sim_start_day="2019-06-03", num_lookback_periods=24 * 60, sim_duration=1
):
    """
    Computes simulation target index for baseload data to be matched on
    :param sim_start_day:
    :param sim_duration:
    :return:
    """

    sim_start_day = pd.to_datetime(sim_start_day).replace(
        hour=0, minute=0, second=0
    )  # ensure set to midnight
    num_lookback_periods_timedelta = pd.to_timedelta(
        num_lookback_periods, unit="minutes"
    )

    # generate index
    target_index_df = list(
        pd.date_range(
            start=pd.to_datetime(sim_start_day) - num_lookback_periods_timedelta,
            periods=(sim_duration * 24 * 60) + num_lookback_periods,
            freq="min",
        )
    )
    target_index_df = pd.DataFrame(target_index_df)
    target_index_df.reset_index(inplace=True)
    target_index_df.columns = ["sim_time", "timestamp"]
    target_index_df["sim_time"] = target_index_df["timestamp"].apply(
        lambda x: time_delta_to_scalar_minutes(x - sim_start_day)
    )
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
    df["ExitDateTime"] = pd.to_datetime(df["ExitDateTime"])
    df["EntryDateTime"] = pd.to_datetime(df["EntryDateTime"])

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


def sample_candidate_days(df, n_days, candidate_days=False, seed=42):
    """
    Sample n=n_days from list of candidate days (with or without replacement) and return df with this subsample
    :param df:
    :param n_days: number of samples to be drawn from candidates, if replace=False this cannot exceed the number of candidates!
    :param candidate_days: list of candidate days in datetime format
    :param replace: whether drawn samples are returned and can be drawn again
    :param seed: seed of random generator, if undefined set to None
    :return:
    """
    if candidate_days:
        if len(candidate_days) < n_days:
            raise ValueError(
                "Number of days to be sampled cannot exceed number of candidate days!!!"
            )
    else:
        candidate_days = df["EntryDate"].unique()  # all days in sample

    # ensure candidate_days are in dt format
    candidate_days_dt = []
    for (
        day
    ) in (
        candidate_days
    ):  # instantiate random number generator and choose days randomly from candidates
        dt = pd.to_datetime(day)
        candidate_days_dt.append(dt)

    rng = default_rng(seed=seed)
    days = rng.choice(
        candidate_days_dt, size=n_days, replace=False
    )  # we do not want to sample the same day twice, therefore set replace=False

    # TODO: Enforce some balance criteria such as same number of winter and summer days, weighted number of weekdays and weekends --> Only relevant for small samples

    df = df[df["EntryDate"].isin(days) == True]

    return df


def select_facilities(df, facility_names):
    """
    Selects subset of preference data that belongs to facilities in facility_names list
    :param df: preference data as DataFrame
    :param facility_names: list of facilities
    :return: DataFrame containing events that belong to facilities in facility_names
    """
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
        carry_over_events_df = pd.DataFrame(columns=df.columns)  # initialize

        for day in df["EntryDate"].unique():

            # select facility and day
            day_df = df[(df["SiteID"] == facility_name) & (df["EntryDate"] == day)]

            # sample events up to max_occupancy
            event_count = round(
                max_occupancy * 2.5
            )  # initialize to 2.5 the max occupancy (seems like a good value)

            while True:
                # sample n=event_count events
                # print(event_count)
                day_df_sampled = day_df.sample(
                    n=event_count, replace=replace, random_state=seed
                )
                day_df_sampled = day_df_sampled.append(carry_over_events_df)

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
                    lg.warning(max_daily_occ)
                # update event count if not max_occupancy not within margin, else break
                if max_daily_occ > round(max_occupancy + (margin * max_occupancy)):
                    if event_count > 0:  # ensure we do not get negative event count!
                        event_count += (
                            -1
                        )  # reduce event counts if above max_occupancy + margin
                    else:
                        lg.warning("Warning! No samples taken for this day!")
                        break
                    if verbose:
                        lg.warning("occupancy too large")
                elif max_daily_occ < round(max_occupancy - (margin * max_occupancy)):
                    event_count += 1  # increase event counts if below max_occupancy
                    if verbose:
                        lg.warning("occupancy too small")
                elif np.absolute(max_daily_occ - max_occupancy) <= round(
                    margin * max_occupancy
                ):
                    lg.warning(
                        "Facility {}; Day {}: occupancy within margin at {} parking events".format(
                            facility_name, day, event_count
                        )
                    )
                    break  # break loop if within margin

            # append to out_df
            out_df = out_df.append(day_df_sampled)
            carry_over_events_df = out_df[out_df["ExitDate"] > day]

        # append facility df to final df
        final_df = final_df.append(out_df)

    return final_df


def sample_up_to_capacity_inc_proportionality_seasonality(
    df, facility_capa=100, margin=0.0015, temp_res=5, seed=42
):
    """
    Samples events for a facility of size capa. Takes into account poportionality of groups (i.e., parker types)
    and seasonality of oarking requests (e.g., lower occupancy on weekends, etc.)
    :param df:
    :param facility_capa:
    :param margin:
    :param temp_res:
    :return:
    """
    scaled_occ = get_scaled_true_occupancy(
        df, agg_level=5
    )  # get true occupancy over horizon and scale

    df["EntryDateTime"] = pd.to_datetime(df["EntryDateTime"])
    df["EntryDate"] = df["EntryDateTime"].apply(lambda x: x.date())
    df["ExitDateTime"] = pd.to_datetime(df["ExitDateTime"])
    df["ExitDate"] = df["ExitDateTime"].apply(lambda x: x.date())
    df["EntryDateTime{}min".format(temp_res)] = df["EntryDateTime"].apply(
        lambda x: x.replace(second=0, minute=round_down_to_base(x.minute, temp_res))
    )
    df["ExitDateTime{}min".format(temp_res)] = df["ExitDateTime"].apply(
        lambda x: x.replace(second=0, minute=round_down_to_base(x.minute, temp_res))
    )

    output_reqs_df = pd.DataFrame()

    for facility in df["SiteID"].unique():

        df_fac = df[df["SiteID"] == facility]

        full_period_sample = pd.DataFrame()
        # full_period_occupancy = pd.DataFrame()
        request_carried_over = pd.DataFrame()
        occupancy_carried_over = len(request_carried_over)

        for day in df_fac["EntryDate"].unique():
            print(day, "started")
            print(occupancy_carried_over, "vehicles carried over from pervious days")
            # lg.warning(day, "started")
            # lg.warning(occupancy_carried_over, "vehicles carried over from pervious days")

            # prep day df
            df_day = df_fac[df_fac["EntryDate"] == day]
            df_day = df_day.append(
                request_carried_over
            )  # append request that remain in hub from previous day

            max_daily_occ = 0
            # print(facility_capa)
            capa_day = round(facility_capa * scaled_occ.loc[day, "scaled_max_occ"])
            n_events = round(capa_day * 3)  # start with same events every day

            loop_timeout = 0
            while True:  # max_daily_occ <= (round(capa + (margin * capa))):

                # sample events
                sample = sample_proportionately(
                    df_day, n=n_events, groups="ClusterName", seed=seed
                )
                # print("sample size:",len(sample))
                sample = sample.append(
                    request_carried_over
                )  # add requests that were carries over

                # get entry counts
                entry_counts = (
                    sample[["EntryDateTime{}min".format(temp_res), "EntryDate"]]
                    .groupby(["EntryDateTime{}min".format(temp_res)])
                    .count()
                )
                entry_counts.reset_index(inplace=True)
                entry_counts.columns = ["DateTime{}min".format(temp_res), "EntryCounts"]
                # print("Entry Counts", entry_counts["EntryCounts"].sum())

                # get exit counts (incl. vehicles that entered day(s) before)
                exit_counts = (
                    sample[["ExitDateTime{}min".format(temp_res), "EntryDate"]]
                    .groupby(["ExitDateTime{}min".format(temp_res)])
                    .count()
                )
                exit_counts.reset_index(inplace=True)
                exit_counts.columns = ["DateTime{}min".format(temp_res), "ExitCounts"]
                # print("Exit Counts", exit_counts["ExitCounts"].sum())

                ## compute target index
                target_index = pd.DataFrame()
                target_index["DateTime{}min".format(temp_res)] = pd.date_range(
                    start=sample["EntryDateTime{}min".format(temp_res)].min(),
                    end=pd.to_datetime(day) + timedelta(hours=23, minutes=59),
                    freq="{}S".format(temp_res * 60),
                )

                # merge
                occupancy = target_index.merge(
                    entry_counts, on="DateTime{}min".format(temp_res), how="left"
                )
                occupancy = occupancy.merge(
                    exit_counts, on="DateTime{}min".format(temp_res), how="left"
                )

                # compute occupancy
                occupancy.fillna(0, inplace=True)
                # occupancy.loc[0,"EntryCounts"] = occupancy.loc[0,"EntryCounts"]+occupancy_carried_over
                occupancy["NetOccupancyChange"] = (
                    occupancy["EntryCounts"] - occupancy["ExitCounts"]
                )
                occupancy["TotalOccupancy"] = occupancy[
                    "NetOccupancyChange"
                ].cumsum()  # gets cumulative sum of occupancy change!

                # limit to day in question
                start = pd.to_datetime(day)
                end = pd.to_datetime(day) + timedelta(hours=23, minutes=59)
                occupancy = occupancy[
                    (occupancy["DateTime{}min".format(temp_res)] >= start)
                    & (occupancy["DateTime{}min".format(temp_res)] <= end)
                ]

                max_daily_occ = max(occupancy["TotalOccupancy"])

                # print("Max occupancy",max_daily_occ,"reached at num events:",n_events)
                if occupancy_carried_over > capa_day:
                    break
                if loop_timeout >= 500:
                    lg.warning("Warning! Loop Timed Out!")
                    break
                elif max_daily_occ == capa_day:
                    lg.warning(
                        "Facility {}; Day {}: max occupancy ({}) within margin (target: {}) at {} parking events ({} iterations)".format(
                            facility,
                            day,
                            max_daily_occ,
                            capa_day,
                            n_events,
                            loop_timeout,
                        )
                    )
                    break  # break loop if within margin
                elif np.absolute(max_daily_occ - capa_day) <= round(margin * capa_day):
                    lg.warning(
                        "Facility {}; Day {}: max occupancy ({})  within margin (target: {}) at {} parking events ({} iterations)".format(
                            facility,
                            day,
                            max_daily_occ,
                            capa_day,
                            n_events,
                            loop_timeout,
                        )
                    )
                    break  # break loop if within margin
                elif max_daily_occ > capa_day:  # round(capa + (margin * capa)):
                    if n_events > 0:  # ensure we do not get negative event count!
                        n_events += (
                            -1
                        )  # reduce event counts if above max_occupancy + margin
                    else:
                        lg.warning("Warning! No samples taken for this day!")
                        break
                elif max_daily_occ < capa_day:  # round(capa + (margin * capa)):
                    n_events += 1  # increase event counts if below max_occupancy

                loop_timeout += 1

            final_day_sample = sample
            request_carried_over = final_day_sample[final_day_sample["ExitDate"] > day]
            request_carried_over.drop(columns=["ExitDate"])
            occupancy_carried_over = len(request_carried_over)

            full_period_sample = full_period_sample.append(final_day_sample)
            # full_period_occupancy = full_period_occupancy.append(occupancy)

        output_reqs_df = output_reqs_df.append(full_period_sample)
        output_reqs_df.sort_values(by=["SiteID", "EntryDateTime"], inplace=True)

    return output_reqs_df


def sample_proportionately(df, n, groups, seed=42):
    """
    Draws a total of n samples fro df but ensures proportionality of groups in resulting sample
    :param df:
    :param n:
    :param groups:
    :return:
    """
    # ratios in original df
    ratios = df.groupby(groups).agg({"EntryDateTime": "count"})
    ratios.rename(columns={"EntryDateTime": "N"}, inplace=True)
    ratios["frac"] = ratios["N"] / len(df)
    df_sampled = pd.DataFrame()
    for group in df[groups].unique():
        sample_size = int(round(ratios.loc[group, "frac"] * n))
        try:
            sample = df[df[groups] == group].sample(
                n=sample_size, replace=False, random_state=seed
            )
        except ValueError:
            sample = df[df[groups] == group].sample(
                n=sample_size, replace=True, random_state=seed
            )
        df_sampled = df_sampled.append(sample)

    return df_sampled


def get_scaled_true_occupancy(requests_df, agg_level=5):
    """
    calculates occupancy from requests and then scales between 0 and 1
    :param requests_df:
    :param agg_level:
    :return:
    """
    requests_df.reset_index(inplace=True)
    dict_rename = {
        "EntryDateTime": "arrival_time",
        "ExitDateTime": "departure_time",
        "EV_yn": "ev_yn",
        "SiteID": "facility",
        "ClusterName": "user_type",
        "index": "vehicle_id",
    }

    requests_df = requests_df.rename(columns=dict_rename)

    # settings
    agg_level = agg_level  # minutes

    # set entry/exit time to target res
    requests_df["arrival_time"] = pd.to_datetime(requests_df["arrival_time"])
    requests_df["departure_time"] = pd.to_datetime(requests_df["departure_time"])
    requests_df["arrival_time"] = requests_df["arrival_time"].apply(
        lambda x: x.replace(second=0, minute=round_down_to_base(x.minute, agg_level))
    )
    requests_df["departure_time"] = requests_df["departure_time"].apply(
        lambda x: x.replace(second=0, minute=round_down_to_base(x.minute, agg_level))
    )  # .apply(lambda x: x.replace(second=0, microsecond=0))

    # count entries and exits per time period in separate df
    entry_counts = requests_df.groupby(
        ["facility", "arrival_time"], as_index=False
    ).agg({"vehicle_id": "count"})
    entry_counts.columns = ["facility", "time", "entry_counts"]
    entry_counts["key"] = entry_counts.apply(
        lambda x: "{}_{}".format(x.facility, x.time), axis=1
    )

    exit_counts = requests_df.groupby(
        ["facility", "departure_time"], as_index=False
    ).agg({"vehicle_id": "count"})
    exit_counts.columns = ["facility", "time", "exit_counts"]
    exit_counts["key"] = exit_counts.apply(
        lambda x: "{}_{}".format(x.facility, x.time), axis=1
    )

    # compute target index
    target_index = pd.DataFrame()
    # requests_df["arrival_time"] = requests_df["arrival_time"].apply(lambda x: x.replace(second=0))
    # requests_df["departure_time"] = requests_df["arrival_time"].apply(lambda x: x.replace(second=0))
    for facility in requests_df["facility"].unique():
        out = pd.DataFrame()
        out["time"] = pd.date_range(
            start=pd.to_datetime(requests_df["EntryDate"].unique()[0]),
            end=pd.to_datetime(requests_df["EntryDate"].unique()[-1])
            + timedelta(hours=23, minutes=59),
            freq="{}S".format(agg_level * 60),
        )
        out["facility"] = facility
        out["key"] = out.apply(lambda x: "{}_{}".format(x.facility, x.time), axis=1)

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
        out_df = occupancy_df[(occupancy_df["facility"] == facility)]
        out_df.sort_values(by="time", inplace=True, ascending=True)
        out_df["total_occupancy"] = out_df["net_occupancy_change"].cumsum()
        occupancy_final = occupancy_final.append(out_df)

    # times
    occupancy_final["date"] = occupancy_final["time"].apply(lambda x: x.date())
    occupancy_final["hour"] = occupancy_final["time"].apply(lambda x: x.hour)
    occupancy_final["minutes_from_midnight"] = occupancy_final["time"].apply(
        lambda x: minutes_from_midnight(x)
    )

    occupancy_norm = occupancy_final.groupby("date").agg({"total_occupancy": "max"})

    max_occ = occupancy_norm["total_occupancy"].max()
    occupancy_norm["scaled_max_occ"] = occupancy_norm["total_occupancy"].apply(
        lambda x: (x) / max_occ
    )

    return occupancy_norm


#### Generate Energy Demand


def generate_charging_demand_data(
    parking_data,
    base_path,
    ev_share,
    min_stay_minutes,
    max_charge_rate,
    home_charging_share,
    geography,
    demand_gen_approach="charging_demand_data",
    seed=42,
):
    """
    Combined routine for matching energy demand per vehicle to parking data
    :param parking_data:
    :param base_path:
    :param ev_share:
    :param min_stay_minutes:
    :param max_charge_rate:
    :param home_charging_share:
    :param seed:
    :return:
    """
    if type(max_charge_rate) == dict:
        max_charge_rate = max_charge_rate["fast"]
    # generate ev pop, charge request, home charging opportunity
    parking_data = generate_electric_vehicles_yn(
        df=parking_data, ev_share=ev_share, seed=seed
    )
    parking_data = generate_charge_request_yn(
        df=parking_data, min_stay_minutes=min_stay_minutes
    )
    parking_data = generate_home_charging_opportunity(
        df=parking_data, home_charging_share=home_charging_share, seed=seed
    )
    parking_data = generate_battery_size(
        df=parking_data, base_path=base_path, geography=geography, seed=seed
    )

    # add kWh demand using 1) coll filtering and 2)
    energy_demand_data = generate_energy_demand_coll_filter_approach(
        df=parking_data, max_charge_rate=max_charge_rate
    )

    location_dict = {
        "CA": "US",
        "DE": "DE",
        "TX": "US",
        "NY": "US",
        "FL": "US",
        "CO": "US",
    }
    energy_demand_data = generate_energy_demand_travel_based_approach(
        df=energy_demand_data,
        base_path=base_path,
        max_charge_rate=max_charge_rate,
        location=location_dict[geography],
        seed=42,
    )

    # create final energy demand column
    if demand_gen_approach == "travel_demand_data":
        energy_demand_data["final_kWhRequested_updated"] = energy_demand_data[
            "final_kWhRequested_travel_demand"
        ]
    elif demand_gen_approach == "charging_demand_data":
        energy_demand_data["final_kWhRequested_updated"] = energy_demand_data[
            "final_kWhRequested_Coll_Filter"
        ]
    else:
        lg.warning(
            "Choose Demand Modeling Approach!!! [travel_demand_data,charging_demand_data]"
        )

    # crucial! Otherwise requests are not in order
    energy_demand_data.sort_values(by="EntryDateTime", inplace=True)
    energy_demand_data.reset_index(drop=True, inplace=True)

    return energy_demand_data


def generate_energy_demand_coll_filter_approach(df, max_charge_rate):
    """
    xxx
    :param df:
    :param max_charge_rate:
    :return:
    """
    if type(max_charge_rate) == dict:
        max_charge_rate = max_charge_rate["fast"]
    # update charge request
    df["final_kWhRequested_Coll_Filter"] = (
        df["userInputs_kWhRequested"] * df["request_charge_yn"] * df["EV_yn"]
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
    df["final_kWhRequested_Coll_Filter"] = df["final_kWhRequested_Coll_Filter"].apply(
        lambda x: round(x, ndigits=2)
    )

    # ensure that energy demand does not exceed typical max battery size (100kWh)
    max_battery_size = 100  # kWh
    df["final_kWhRequested_Coll_Filter"] = df["final_kWhRequested_Coll_Filter"].apply(
        lambda x: max_battery_size if x > max_battery_size else x
    )

    return df


def generate_energy_demand_travel_based_approach(
    df, base_path, max_charge_rate, location="US", seed=42
):
    """
    xxx
    :param df:
    :param base_path:
    :param max_charge_rate:
    :param location:
    :param seed:
    :return:
    """
    if type(max_charge_rate) == dict:
        max_charge_rate = max_charge_rate["fast"]
    df = generate_daily_distance_traveled(
        df, base_path=base_path, location=location, seed=seed
    )
    # energy per km
    # kWh_per_km = 18.5 / 100  # Tesal Model S75
    kWh_per_km = round(0.2306631886256587, 3)  # mean of ACN sample

    # base charging demand'
    df["final_kWhRequested_travel_demand"] = df.apply(
        lambda x: (
            x.daily_travel_km * kWh_per_km
            if x.request_charge_yn == 1 and x.EV_yn == 1
            else 0
        ),
        axis=1,
    )

    # ensure realistic charge request (i.e., fulfillment physically possible at max charge rate)
    df["final_kWhRequested_travel_demand"] = df.apply(
        lambda x: (
            x.MinutesStay / 60 * max_charge_rate
            if x.final_kWhRequested_travel_demand
            >= x.MinutesStay / 60 * max_charge_rate
            else x.final_kWhRequested_travel_demand
        ),
        axis=1,
    )
    df["final_kWhRequested_travel_demand"] = df[
        "final_kWhRequested_travel_demand"
    ].apply(lambda x: round(x, ndigits=2))

    # ensure that energy demand does not exceed typical max battery size (100kWh)
    max_battery_size = 100  # kWh
    df["final_kWhRequested_travel_demand"] = df[
        "final_kWhRequested_travel_demand"
    ].apply(lambda x: max_battery_size if x > max_battery_size else x)

    return df


def generate_electric_vehicles_yn(df, ev_share, seed=42):
    """
    Selects whether parked vehicle is an EV with probability p=ev_share (Brenoulli Process). Sets energy demand of vehicles
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

    return df


def generate_charge_request_yn(df, min_stay_minutes=20):
    """
    Defines whether vehicle requests charging or not. This is implemented as a minimum stay duration threshold currently
    :param df:
    :param ev_share: [0,1]
    :param seed: seed of random generator, if undefined set to None
    :return: adds colums EV_yn and "final_kWhRequested_updated"
    """

    # TODO: May be modeles as Conditional Expectation of Success given Planned Duration of Stay
    # For now: assume minimum stay duration of XX
    df["request_charge_yn"] = df.apply(
        lambda x: 1 if x.EV_yn == 1 and x.MinutesStay >= min_stay_minutes else 0, axis=1
    )

    return df


def generate_daily_distance_traveled(df, base_path, location="US", seed=42):
    us_mean_km = 25.9 * 1.60934  # https://tedb.ornl.gov/data/ Table 9.09
    de_mean_km = (
        (1051 + 1149) / 2 / 30
    )  # https://mobilitaetspanel.ifv.kit.edu/english/downloads.php; 2019 vehicle miles

    if location == "US":
        # get data and prepare
        file_loc = "EV_Energy_Demand_Data/CA_Travel_survey_trip.csv"
        travel_demand = pd.read_csv(os.path.join(base_path, file_loc))
        travel_demand = travel_demand[
            [
                "sampno",
                "perno",
                "tripno",
                "trpmiles",
                "distance_mi",
                "trvlcmin",
                "trptrans17",
            ]
        ]
        # limit to relevant transport modes only
        travel_demand_cars = travel_demand[
            travel_demand["trptrans17"].isin([3, 4, 6, 5, 18]) == True
        ]
        # group by sample and process
        travel_demand_cars_grouped = travel_demand_cars.groupby("sampno").sum()
        # remove unreasonably large numbers
        travel_demand_cars_grouped = travel_demand_cars_grouped[
            travel_demand_cars_grouped["distance_mi"]
            <= travel_demand_cars_grouped["distance_mi"].quantile(q=0.99)
        ]
        # remove unreasonably small numbers (likely roundtrips)
        travel_demand_cars_grouped = travel_demand_cars_grouped[
            travel_demand_cars_grouped["distance_mi"] > 0.2
        ]

        # sample randomly with repalcement (need to fit an empirical distribution here!)
        distances = travel_demand_cars_grouped.sample(
            n=len(df), replace=True, random_state=seed
        )
        distances = distances["distance_mi"] * 1.60934
        distances = np.array(distances)

        # distances = rng.exponential(us_mean_km, size=len(df))

    elif location == "DE":
        rng = default_rng(seed=seed)
        # We only have the mean here: --> assume exponential distribution
        # It can be shown for the exponential distribution that the mean is equal to the standard deviation; i.e., μ = σ = 1/λ
        distances = rng.exponential(de_mean_km, size=len(df))

    df["daily_travel_km"] = distances

    return df


def generate_home_charging_opportunity(df, home_charging_share, seed=42):
    """
    Brenoulli Process that assigns home charging with probability p(HomeCharging) = home_charging_share
    :param df:
    :param home_charging_share:
    :param seed:
    :return:
    """

    rng = default_rng(seed=seed)
    home_charging_array = rng.choice(
        [0, 1], p=[1 - home_charging_share, home_charging_share], size=len(df)
    )
    df["home_charging_yn"] = home_charging_array

    return df


def generate_battery_size(df, base_path, geography="CA", seed=42):
    """
    Samples battery size based on empirical distribution of EV stock in selected geography
    :param df:
    :param geography: select from DE, CA, TX, NY, FL, CO --> This is for choosing which empirical distribution of vehicle stock to use
    :return: Column "BatterySize"
    """

    file_name = "EV_Stock_Data/EV_Stock_Data.xlsx"

    # get batteries
    batteries = pd.read_excel(os.path.join(base_path, file_name), engine="openpyxl")  #
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
    rng = default_rng(seed=seed)
    capacity = batteries["Battery Capacity Combined (in kWh)"]
    probability = batteries["Vehicle_probability"]

    df["BatterySize"] = df["EV_yn"].apply(
        lambda x: (
            rng.choice(capacity, p=probability, replace=False) if x == 1 else np.NaN
        )
    )

    return df


#################################################################################################
#################################################################################################
############ ALTERNATIVE APPROACH: Sample each vehicle independently (prior or during simulation)
# This is a three-step process using empirical distributions
# Good introductory resource is here: https://machinelearningmastery.com/empirical-distribution-function-in-python/


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


def round_down_to_base(num, base):
    return num - (num % base)


def minutes_from_midnight(dt):
    mfm = dt.hour * 60 + dt.minute
    return mfm


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
