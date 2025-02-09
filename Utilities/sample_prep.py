# Executes full simulation routine
import Utilities.sim_input_processing as prep
import pandas as pd

DATA_PATH_WS = "/home/kschroer/Documents/Research/EV_Charging_Clusters/Data/"
CACHE_PATH_WS = "/home/kschroer/Documents/Research/EV_Charging_Clusters/Cache/"
SAMPLE_PATH_WS = "/home/kschroer/Documents/Research/EV_Charging_Clusters/Samples/"

# PARAMS
SIM_SEASON = ["Summer", "Winter"]
# "Facility_1": "Mixed-use_Facility", "Facility_2": "Hospital", "Facility_3": "Mixed-use_Facility", "Facility_4": "Destination_Facility","Facility_5": "Workplace_Facility", "Facility_6": "Workplace_Facility","Facility_KoeBogen": "Destination_Facility"
FACILITIES = [
    "Facility_KoeBogen"
]  # ["Facility_KoeBogen","Facility_3","Facility_4","Facility_5","Facility_6"]
SIM_START_DAY = pd.to_datetime("2019-01-01")
SIM_DURATION = 365
REGION = "CA"
MAX_CHARGE_RATE = 50
EV_SHARE = 1
NUM_PARKING_SPOTS = [200, 150]  # [100,150,200,250] 100!


for FAC in FACILITIES:
    print(FAC, "started...")
    for CAPA in NUM_PARKING_SPOTS:
        print(FAC, CAPA, "started...")
        df = prep.get_sim_charging_requests(
            base_path=DATA_PATH_WS,
            parking_capacity=CAPA,
            cache_path=CACHE_PATH_WS,
            sim_seasons=SIM_SEASON,
            sim_start_day=SIM_START_DAY,
            sim_duration=SIM_DURATION,
            facility_list=[FAC],
            ev_share=EV_SHARE,
            geography=REGION,
            limit_requests_to_capa=True,
            max_charge_rate=MAX_CHARGE_RATE,
        )

        df.sort_values(by="EntryDateTime", inplace=True)
        df.reset_index(drop=True)
        try:
            df.drop(
                columns=[
                    "EntryMinutesFromSimStart",
                    "ExitMinutesFromSimStart",
                    "EntryDateTime5min",
                    "ExitDateTime5min",
                ],
                inplace=True,
            )
        except:
            "not in columns"
        df.to_pickle(
            CACHE_PATH_WS
            + "requests_sample_{}_size_limit_{}_max_charge_rate_{}.pkl".format(
                FAC, CAPA, MAX_CHARGE_RATE
            )
        )
        # df.to_csv(SAMPLE_PATH_WS + "requests_sample_{}_size_limit_{}.csv".format(FAC, CAPA))
