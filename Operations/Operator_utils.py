# utilities.py

def get_exp_free_grid_capacity_utility(
    current_time,
    sim_time,
    planning_interval,
    num_lookahead_planning_periods,
    num_lookback_periods,
    baseload,
    non_dispatchable_generator,
    electric_storage,
    charging_strategy,
    charging_hub,
    grid_capa
):
    final_time = min(
        sim_time,
        round(current_time + planning_interval * num_lookahead_planning_periods),
    )

    periods = []
    t = current_time
    while t < final_time:
        periods.append(t)
        t += planning_interval

    free_capa_list_actual = []
    base_load_list = []
    generation_list = []
    free_capa_list_predicted = []

    for t in periods:
        # ACTUAL
        baseload_max = max(
            baseload.loc[t : t + planning_interval - 1]["load_kw_rescaled"]
        )
        generation_min = min(
            non_dispatchable_generator.generation_profile_actual.loc[
                t : t + planning_interval - 1
            ]["pv_generation"]
        )
        battery_max = min(
            electric_storage.kW_discharge_peak,
            (
                (
                    electric_storage.SoC
                    - electric_storage.min_energy_stored_kWh
                )
                * (60 / planning_interval)
            ),
        )
        if charging_strategy in [
            "dynamic",
            "integrated_storage",
            "online_multi_period",
        ]:
            battery_max = 0

        battery_usage = 0
        if charging_hub.dynamic_pricing:
            battery_usage = (
                electric_storage.discharging_power
                - electric_storage.charging_power
            )

        free_capa_list_actual.append(
            grid_capa
            - baseload_max
            + generation_min
            + battery_max
            + battery_usage
        )

        # PREDICTED
        offset_period = num_lookback_periods
        baseload_max_pred = max(
            baseload.loc[
                (t - offset_period):(t - offset_period) + (planning_interval - 1)
            ]["load_kw_rescaled"]
        )
        generation_min_pred = min(
            non_dispatchable_generator.generation_profile_forecast.loc[
                t : t + planning_interval - 1
            ]["pv_generation"]
        )
        battery_max = min(
            electric_storage.kW_discharge_peak,
            (
                (
                    electric_storage.SoC
                    - electric_storage.min_energy_stored_kWh
                )
                * (60 / planning_interval)
            ),
        )

        free_capa_list_predicted.append(
            grid_capa
            - baseload_max_pred
            + generation_min_pred
            + battery_max
            + battery_usage
        )

        base_load_list.append(baseload_max_pred)
        generation_list.append(generation_min_pred)

    free_grid_capa_without_storage = (
        free_capa_list_actual[0] - battery_max - battery_usage
    )

    return {
        "free_grid_capa_actual": free_capa_list_actual,
        "base_load_list": base_load_list,
        "generation_list": generation_list,
        "free_grid_capa_without_storage": free_grid_capa_without_storage,
        "free_grid_capa_predicted": free_capa_list_predicted,
    }