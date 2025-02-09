import simpy

from Environment.helper.configuration.configuration import Configuration


class GridCapacity:

    def __init__(self, env, grid_capa):
        self.env = env
        self.max_capacity = grid_capa
        self.current_usage = 0
        self.grid_usage = []
        self.energy_costs = 0
        self.energy_rewards = 0
        self.B2G = Configuration.instance().B2G

    def update_usage(
        self, base_load, chargers, PV, storage, vehicles, exclude_baseload=False
    ):
        current_usage = 0
        PV_load = PV.generation_profile_actual
        for charger in chargers:
            current_usage += charger.current_power_usage
        current_usage += (
            base_load.loc[self.env.now, "load_kw_rescaled"]
            + storage.charging_power
            - storage.discharging_power
            - PV_load.loc[self.env.now, "pv_generation"]
        )
        if self.B2G:  # We can feed the grid if we have battery to grid possibility
            self.current_usage = current_usage
        else:
            self.current_usage = max(current_usage, 0)
            self.current_usage_without_baseload = max(
                self.current_usage - base_load.loc[self.env.now, "load_kw_rescaled"], 0
            )

    def use(self, charger):
        """
        Assign required grid capacity to a charger
        """
        with self.available_grid_capacity.get(
            charger.power
        ) as get:  # TODO: energy vs. power as resource?
            yield get

    def reset(self, planning_period_length):
        """
        Reset to full capacity (typically at start of new planning period)
        """
        if (
            self.available_grid_capacity.level < self.max_capacity
        ):  # reset if low capacity level
            reset_amount = self.max_capacity - self.available_grid_capacity
            yield self.available_grid_capacity.put(reset_amount)

        yield self.env.timeout(
            planning_period_length
        )  # reset every n=period length periods to ensure full capacity is available at start of planning

    def monitor(self, base_load, chargers, PV, storage, energy_costs, vehicles):
        """
        Monitoring function to track usage of grid capacity in each sim period
        :return:
        """
        while True:

            for charger in chargers:
                charger.status_update()
            self.update_usage(
                base_load=base_load,
                chargers=chargers,
                PV=PV,
                storage=storage,
                vehicles=vehicles,
            )
            self.grid_usage.append(self.current_usage)
            hour = int((self.env.now % 1440 - self.env.now % 60) / 60)
            self.energy_costs += (
                self.current_usage_without_baseload / 60 * energy_costs[hour]
            )
            self.energy_rewards += self.current_usage / 60 * energy_costs[hour]
            yield self.env.timeout(1)

    def reset_reward(self):
        self.energy_rewards = 0
