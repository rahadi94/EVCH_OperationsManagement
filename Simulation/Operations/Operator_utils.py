

def compute_free_grid_capacity(self) -> dict:
    """
    Compute free grid capacity based on expected base load and non-dispatchable generation.
    
    Args:
        self: Operator instance containing all necessary attributes
        
    Returns:
        dict: Dictionary containing computed capacity values and time series
    """
    final_time = min(
        self.sim_time,
        round(self.env.now + self.planning_interval * self.num_lookahead_planning_periods),
    )

    periods = []
    t = self.env.now
    while t < final_time:
        periods.append(t)
        t += self.planning_interval

    free_capa_list_actual = []
    base_load_list = []
    generation_list = []
    free_capa_list_predicted = []
    first_period_without_storage = None

    for t in periods:
        # ACTUAL
        baseload_max = max(
            self.baseload.loc[t : t + self.planning_interval - 1]["load_kw_rescaled"]
        )
        generation_min = min(
            self.non_dispatchable_generator.generation_profile_actual.loc[
                t : t + self.planning_interval - 1
            ]["pv_generation"]
        )
        battery_max = min(
            self.electric_storage.kW_discharge_peak,
            (
                (
                    self.electric_storage.SoC
                    - self.electric_storage.min_energy_stored_kWh
                )
                * (60 / self.planning_interval)
            ),
        )
        if self.charging_strategy in [
            "dynamic",
            "integrated_storage",
            "online_multi_period",
        ]:
            battery_max = 0

        battery_usage = 0
        if self.charging_hub.dynamic_pricing:
            battery_usage = (
                self.electric_storage.discharging_power
                - self.electric_storage.charging_power
            )

        free_capa_list_actual.append(
            self.grid_capa
            - baseload_max
            + generation_min
            + battery_max
            + battery_usage
        )

        # Capture the first period's capacity without storage correctly (no battery discharge or usage)
        if first_period_without_storage is None:
            first_period_without_storage = self.grid_capa - baseload_max + generation_min

        # PREDICTED
        offset_period = self.num_lookback_periods
        baseload_max_pred = max(
            self.baseload.loc[
                (t - offset_period):(t - offset_period) + (self.planning_interval - 1)
            ]["load_kw_rescaled"]
        )
        generation_min_pred = min(
            self.non_dispatchable_generator.generation_profile_forecast.loc[
                t : t + self.planning_interval - 1
            ]["pv_generation"]
        )
        battery_max = min(
            self.electric_storage.kW_discharge_peak,
            (
                (
                    self.electric_storage.SoC
                    - self.electric_storage.min_energy_stored_kWh
                )
                * (60 / self.planning_interval)
            ),
        )

        free_capa_list_predicted.append(
            self.grid_capa
            - baseload_max_pred
            + generation_min_pred
            + battery_max
            + battery_usage
        )

        base_load_list.append(baseload_max_pred)
        generation_list.append(generation_min_pred)

    # Use the first period's actual baseload and generation only
    free_grid_capa_without_storage = first_period_without_storage if first_period_without_storage is not None else 0

    return {
        "free_grid_capa_actual": free_capa_list_actual,
        "base_load_list": base_load_list,
        "generation_list": generation_list,
        "free_grid_capa_without_storage": free_grid_capa_without_storage,
        "free_grid_capa_predicted": free_capa_list_predicted,
    }