import pandas as pd
import simpy
import Utilities.sim_input_processing as prep

from Infrastructure.electric_generator import NonDispatchableGenerator

env = simpy.Environment()


def data_preparation(facility, date, t_0, size=200):
    # random.seed(42)
    data_path = "/home/rahadi/Projects/EVCC/kshroer/EV_Charging_Clusters/Samples/"
    raw_input_path = "/home/rahadi/Projects/EVCC/kshroer/EV_Charging_Clusters/Data/"
    cache_path = "/home/rahadi/Projects/EVCC/kshroer/EV_Charging_Clusters/Cache/"
    pv_gen = NonDispatchableGenerator(
        env=env,
        kW_peak=1,
        base_path=raw_input_path,
        sim_start_day=date,
        sim_duration=1,
        num_lookback_periods=24,
        cache_path=cache_path,
    ).get_forecast_generation_profile()
    pv_profile = (
        pv_gen[pv_gen["day"] == int(date[-2] + date[-1])]
        .groupby("hour")
        .agg({"pv_generation": "mean"})
    )
    pv_profile = pv_profile["pv_generation"].values
    # loads = pd.read_csv('Planning/Scaled_Avg_Hourly_Load.csv')
    # loads = loads.loc[(loads['SiteID'] == 'Facility_KoeBogen')].groupby('hour').agg({'kWScaled': 'mean'})

    loads = prep.get_sim_baseload_curve(
        base_path=raw_input_path,
        cache_path=cache_path,
        sim_start_day=date,
        num_lookback_periods=0,
        sim_duration=1,
        min_facility_baseload=220,
        max_facility_baseload=73,
    )["load_kw_rescaled"]

    df = pd.read_csv(
        f"/home/rahadi/Projects/EVCC/kshroer/EV_Charging_Clusters/Samples/requests_sample_{facility}_size_limit_{size}_max_charge_rate_50.csv"
    )
    df1 = df[
        (df["EntryDate"] == date)
        & (df["EntryDate"] == df["ExitDate"])
        & (df["HoursStay"] >= 0)
        & (df["EntryHour"] >= 1)
        & (df["ExitHour"] <= 23)
    ]
    return df1, loads, pv_profile
    # number = 400
    # while True:
    #     dfs = df1.sample(n=number, random_state=1)
    #     occupation = pd.DataFrame(range(24))
    #     for i in range(24):
    #         occupation.iloc[i][0] = dfs[(dfs['EntryHour'] <= i) & (dfs['ExitHour'] >= i)]['EntryHour'].count()
    #     if occupation[0].max() <= 100:
    #
    #         return dfs, loads, pv_profile
    #     number -= 1
    #     # print(number)
    # dfs = df1.sample(n=500, weights=df.groupby('ClusterNum')['ClusterNum'].transform('count'))
