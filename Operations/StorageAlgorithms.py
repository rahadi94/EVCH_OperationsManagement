# Contains algorithms for battery storage operations
from Environment.log import lg
import numpy as np


def uncontrolled(env, storage_object):
    """
    Algo for mock-implementation of battery. Decides whether to charge or discharge first, then sets power randomly,
    neglects infrastructure constraints, including battery SoC
    :return:
    """
    # reset charging state
    storage_object.charge_yn = 0
    storage_object.discharge_yn = 0

    # reset charging power
    storage_object.charging_power = 0
    storage_object.discharging_power = 0

    # decide if to charge or discharge on a random basis --> ensures that you cannot do both at the same time
    charge_dec = np.random.randint(-1, 2, 1)[0]

    # update charge states
    if charge_dec == 1:
        storage_object.charge_yn = 1
    elif charge_dec == -1:
        storage_object.discharge_yn = 1
    elif charge_dec == 0:
        pass

    # pick a random rate and
    rate = np.random.randint(1, 150, 1)[0]  # in kW
    if storage_object.charge_yn == 1:
        storage_object.charging_power = rate
        lg.info(f"Storage charging at rate {rate}kW for planning period {env.now}")
    elif storage_object.discharge_yn == 1:
        storage_object.discharging_power = rate
        lg.info(f"Storage discharging at rate {rate}kW for planning period {env.now}")
    else:
        lg.info(f"Storage idle for planning period {env.now}")


def temporal_arbitrage(
    env,
    storage_object,
    planning_interval,
    electricity_tariff,
    free_grid_capacity,
    ev_charging_load,
):
    """
    - Checks if battery is requested by EV charging algo, if yes, provide this energy,
    - else:
        - Discharge equally during on-peak
        - Charge equally during off-peak periods until next on-peak
    Loosely based on Gust et al. (2021,EJOR)
    :return:
    """

    # reset
    storage_object.charge_yn = 0
    storage_object.discharge_yn = 0
    storage_object.charging_power = 0
    storage_object.discharging_power = 0

    sim_period = env.now
    tariff_state = get_tariff_state(
        sim_period=sim_period, electricity_tariff=electricity_tariff
    )

    if type(free_grid_capacity) == list:
        free_grid_capacity = free_grid_capacity[0]

    if ev_charging_load > free_grid_capacity:  # use battery to fulfill charging needs

        storage_object.discharge_yn = 1
        storage_object.charge_yn = 0
        storage_object.charging_power = 0

        # set rate
        # max_remaining_charge = storage_object.max_energy_stored_kWh - storage_object.SoC
        max_remaining_discharge = (
            storage_object.SoC - storage_object.min_energy_stored_kWh
        )

        if storage_object.discharge_yn == 1:
            storage_object.discharging_power = min(
                ev_charging_load - free_grid_capacity,
                (max_remaining_discharge * (60 / planning_interval)),
                storage_object.kW_discharge_peak,
            )
            lg.info(
                f"Storage discharging at rate {storage_object.discharging_power}kW for planning period starting {env.now}"
            )

    else:  # run temporal arbitrage

        # charge or discharge:
        if tariff_state == "offpeak":
            storage_object.charge_yn = 1
            storage_object.discharge_yn = 0
        if tariff_state == "onpeak":
            storage_object.charge_yn = 0
            storage_object.discharge_yn = 1

        # get remaining periods until state change
        remaining_periods_current_state = 1
        while True:
            if get_tariff_state(
                sim_period=(sim_period + remaining_periods_current_state),
                electricity_tariff=electricity_tariff,
            ) == get_tariff_state(
                sim_period=(sim_period + remaining_periods_current_state + 1),
                electricity_tariff=electricity_tariff,
            ):  # if same as next state
                remaining_periods_current_state += 1
            else:
                remaining_periods_current_state = min(
                    remaining_periods_current_state, planning_interval
                )
                break

        # set rate
        max_remaining_charge = storage_object.max_energy_stored_kWh - storage_object.SoC
        max_remaining_discharge = (
            storage_object.SoC - storage_object.min_energy_stored_kWh
        )
        if storage_object.charge_yn == 1:
            storage_object.charging_power = min(
                max_remaining_charge / (remaining_periods_current_state / 60),
                storage_object.kW_charge_peak,
            )
            lg.info(
                f"Storage charging at rate {storage_object.charging_power}kW for planning period {env.now}"
            )
        if storage_object.discharge_yn == 1:
            storage_object.discharging_power = min(
                max_remaining_discharge / (remaining_periods_current_state / 60),
                storage_object.kW_discharge_peak,
            )
            lg.info(
                f"Storage discharging at rate {storage_object.discharging_power}kW for planning period {env.now}"
            )


def peak_shaving(
    env,
    storage_object,
    planning_interval,
    free_grid_capacity,
    ev_charging_load,
    max_base_load,
    min_PV_generation,
    peak_history_inc_storage,
):
    """
    - Checks if battery is requested by EV charging algo, if yes, provide this energy,
    - else:
        - Discharge at full required capa if threshold is exceeded (or if peak is expected)
        - Charge at full capacity to replenish while considering peaks and other constraints
    Loosely based on Gust et al. (2021,EJOR)
    :return:
    """

    exp_peak = ev_charging_load + max_base_load - min_PV_generation
    if len(peak_history_inc_storage) > 1:
        historic_peak = max(peak_history_inc_storage[0:-1])
    else:
        historic_peak = max(peak_history_inc_storage)

    # reset
    storage_object.charge_yn = 0
    storage_object.discharge_yn = 0
    storage_object.charging_power = 0
    storage_object.discharging_power = 0

    if type(free_grid_capacity) == list:
        free_grid_capacity = free_grid_capacity[0]

    if (
        ev_charging_load > free_grid_capacity
    ):  # use battery to fulfill charging needs if charging algo requests battery

        storage_object.discharge_yn = 1
        storage_object.charge_yn = 0
        storage_object.charging_power = 0

        # set rate
        max_remaining_discharge = (
            storage_object.SoC - storage_object.min_energy_stored_kWh
        )

        if storage_object.discharge_yn == 1:
            storage_object.discharging_power = min(
                ev_charging_load - free_grid_capacity,
                (max_remaining_discharge * (60 / planning_interval)),
                storage_object.kW_discharge_peak,
            )
            lg.info(
                f"Storage discharging at rate {storage_object.discharging_power}kW for planning period starting {env.now}"
            )

    else:  # run peak shaving routine, if battery otherwise not used

        # print("Exp. peak", exp_peak)
        # print("Hist. peak", historic_peak)
        if exp_peak > historic_peak:  # set discharging
            storage_object.charge_yn = 0
            storage_object.discharge_yn = 1
        elif exp_peak < historic_peak:  # set charging
            storage_object.charge_yn = 1
            storage_object.discharge_yn = 0
        else:
            storage_object.charge_yn = 0
            storage_object.discharge_yn = 0

        # print("discharge:", storage_object.discharge_yn)
        # print("charge:",storage_object.charge_yn)

        # set charge/discharge rates...
        max_remaining_charge = storage_object.max_energy_stored_kWh - storage_object.SoC
        max_remaining_discharge = (
            storage_object.SoC - storage_object.min_energy_stored_kWh
        )

        # discharge, i.e., peak shaving
        if storage_object.discharge_yn == 1:
            peak_to_shave = exp_peak - historic_peak
            storage_object.discharging_power = min(
                peak_to_shave,
                (max_remaining_discharge * (60 / planning_interval)),
                storage_object.kW_discharge_peak,
            )
            lg.info(
                f"Storage discharging at rate {storage_object.discharging_power}kW for planning period {env.now}"
            )

        # charge, i.e., refill for later peak shaving
        if storage_object.charge_yn == 1:
            peak_not_to_exceed = (
                historic_peak - exp_peak
            )  # this is what is available for battery dicharging while avoiding a new peak!
            storage_object.charging_power = min(
                peak_not_to_exceed,
                max_remaining_charge * (60 / planning_interval),
                storage_object.kW_charge_peak,
            )
            lg.info(
                f"Storage charging at rate {storage_object.charging_power}kW for planning period {env.now}"
            )

        # print("discharge rate:", storage_object.discharging_power)
        # print("charge rate:", storage_object.charging_power)


######HELPER########
def get_tariff_state(sim_period, electricity_tariff):
    """
    Identifies tariff regime of current sim_period
    :param sim_period:
    :param electricity_tariff:
    :return:
    """

    hour = int(sim_period % 1440 / 60)

    if electricity_tariff[hour] == max(electricity_tariff):
        state = "onpeak"
    elif electricity_tariff[hour] == min(electricity_tariff):
        state = "offpeak"

    return state
