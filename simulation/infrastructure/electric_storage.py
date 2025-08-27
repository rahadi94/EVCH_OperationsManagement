class ElectricStorage:

    def __init__(self, env, max_capacity_kWh, min_capacity_kWh=0, efficiency=1):
        self.env = env
        self.info = {}  # information for final results
        self.storage_type = "LiIon-Storage"  # provide details on type of storage
        self.charge_yn = 0  # binary indicator of whether storage is being charged
        self.charge_yn_plan = dict()
        self.discharge_yn = 0  # binary indicator of whether storage is being discharged
        self.discharge_yn_plan = dict()
        self.mode = None  # "Charging", "Discharging", "Idle"
        self.charging_power = 0
        self.charging_power_plan = dict()
        self.discharging_power = 0
        self.discharging_power_plan = dict()
        self.max_energy_stored_kWh = max_capacity_kWh
        self.min_energy_stored_kWh = min_capacity_kWh
        self.kW_charge_peak = (
            1 * max_capacity_kWh
        )  # factor taken from EJOR microgrid paper (Gust et al.), for now assume symmetric charge/discharge capacity
        self.kW_discharge_peak = (
            1 * max_capacity_kWh
        )  # factor taken from EJOR microgrid paper (Gust et al.), for now assume symmetric charge/discharge capacity
        self.efficiency = efficiency  # assume symmetric efficiency
        self.SoC = 0.5 * max_capacity_kWh  # kWh
        self.storage_health = (
            100  # may use this to account for degradation (need a degradation model!)
        )

    def deploy(
        self, B2G=False, hub_demand_kW=0, hub_generation_kW=0, max_grid_capa=1000
    ):

        # set mode
        if self.charge_yn == 1:
            self.mode = "Charging"
        elif self.discharge_yn == 1:
            self.mode = "Discharging"
        else:
            self.mode = "Idle"
        # charge rate cannot exceed Soc and grid capa

        if self.SoC > self.max_energy_stored_kWh:
            self.charging_power = 0
        if self.charge_yn == 1:
            if self.SoC + self.charging_power / 60 > self.max_energy_stored_kWh:
                self.charging_power = (self.max_energy_stored_kWh - self.SoC) * 60
            # if charging_algo not in ['dynamic']:
            if (
                max_grid_capa - hub_demand_kW + hub_generation_kW - self.charging_power
                < 0
            ):
                self.charging_power = max_grid_capa - hub_demand_kW + hub_generation_kW
            self.charging_power = max((self.charging_power), 0)
        # discharge rate cannot exceed SoC, and hub demand (i.e., no infeed)
        if self.discharge_yn == 1:
            if self.SoC - self.discharging_power / 60 < 0:
                self.discharging_power = max((self.SoC) * 60, 0)
            if not B2G:
                if (
                    self.discharging_power > hub_demand_kW - hub_generation_kW
                ):  # check if discharge would exceed net demand (excl. grid)
                    self.discharging_power = (
                        hub_demand_kW - hub_generation_kW
                    )  # set to net demand
            self.discharging_power = max((self.discharging_power), 0)
        # update state of charge
        self.SoC += (
            min(self.kW_charge_peak, self.charging_power * self.charge_yn) / 60
        ) * self.efficiency
        self.SoC += (
            max(-self.kW_discharge_peak, -(self.discharging_power * self.discharge_yn))
            / 60
        ) * self.efficiency

    def update_health(self):
        # TODO: battery ramping updates kWh_capacity
        pass

    def monitor(self):
        """
        Monitoring the status of chargers (number of connected and charging vehicles) every time step
        """
        self.info["SoC"] = []
        self.info["mode"] = []
        self.info["charge_load"] = []
        self.info["discharge_load"] = []

        while True:
            self.info["SoC"].append(self.SoC)
            self.info["mode"].append(self.mode)
            self.info["charge_load"].append(self.charging_power)
            self.info["discharge_load"].append(self.discharging_power)

            yield self.env.timeout(1)
