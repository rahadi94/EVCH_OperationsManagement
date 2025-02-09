import pandas as pd

import simpy
import logging
from Environment.Initialisation import (
    SIM_TIME,
    NUM_PARKING_SPOTS,
    GRID_CAPA,
    CHARGER_CAPA,
    sim_data,
    MAX_NUM_CONNECTORS,
    ELECTRICITY_TARIFF,
)
from Environment.model import EVCC_Sim_Model
from Environment.log import stream_handler

stream_handler.setLevel(logging.ERROR)
list_NUM = []
CHARGER_NUM = 1000
iteration = 0
service_level = []
condition = True
direction = "backward"
# TODO: Considering the stochastic demand (using expected service level?)
# TODO: Changing the stop condition of the algorithm
while condition:
    print(f"Number of chargers={CHARGER_NUM}")
    env = simpy.Environment()  # Creating the simpy environment
    model = EVCC_Sim_Model(
        env=env,
        sim_time=SIM_TIME,
        parking_capa=NUM_PARKING_SPOTS,
        grid_capa=GRID_CAPA,
        charging_capa=CHARGER_CAPA,
        charging_num=CHARGER_NUM,
        connector_num=MAX_NUM_CONNECTORS,
        request_data=sim_data,
        electricity_tariff=ELECTRICITY_TARIFF,
    )
    model.run()
    env.run(until=model.sim_time)
    total_energy_requested = 0
    total_energy_charged = 0
    for request in model.requests:
        total_energy_requested += request.energy_requested
        total_energy_charged += request.energy_charged
    service_level.append(total_energy_charged / total_energy_requested)
    list_NUM.append(CHARGER_NUM)
    print(service_level)
    print(list_NUM)
    if iteration == 0:
        CHARGER_NUM = int(round(CHARGER_NUM / 2))
    if iteration >= 1:
        if service_level[0] - service_level[iteration] < 0.05:
            if direction == "backward":
                CHARGER_NUM = int(round(CHARGER_NUM / 2))
            if direction == "forward":
                target = list_NUM[-2]
                CHARGER_NUM = int(round((list_NUM[-1] + target) / 2))
        else:
            direction = "forward"
            list_NUM.remove(CHARGER_NUM)
            target = list_NUM[-1]
            CHARGER_NUM = int(round((CHARGER_NUM + target) / 2))
    print(f"service_level = {total_energy_charged / total_energy_requested}")
    if iteration >= 2:
        if list_NUM[-1] == list_NUM[-2]:
            condition = False
    iteration += 1
