import Utilities.sim_input_processing as prep

# Subclass: Dispatchable
# Subclass: NonDispatchable


class ElectricGenerator:

    def __init__(
        self,
        env,
        kW_peak,
        base_path,
        cache_path,
        sim_start_day,
        sim_duration,
        num_lookback_periods,
    ):
        self.env = env
        self.kW_peak = kW_peak
        self.base_path = base_path
        self.cache_path = cache_path
        self.sim_start_day = sim_start_day
        self.sim_duration = sim_duration
        self.num_lookback_periods = num_lookback_periods
        # ....


class NonDispatchableGenerator(ElectricGenerator):
    """
    This is a subclass of the ElectricGenerator class
    """

    def __init__(
        self,
        env,
        kW_peak,
        base_path,
        cache_path,
        sim_start_day,
        sim_duration,
        num_lookback_periods,
        generator_type="rooftop_PV",
        unit_name="PV1",
    ):
        ElectricGenerator.__init__(
            self,
            env,
            kW_peak,
            base_path,
            cache_path,
            sim_start_day,
            sim_duration,
            num_lookback_periods,
        )
        self.generator_type = generator_type  # rooftop PV, wind, etc.
        self.unit_name = unit_name  # PV

        # variable load factor per each sim period
        self.generation_profile_actual = self.get_actual_generation_profile()
        self.generation_profile_forecast = self.get_forecast_generation_profile()

        # TODO: Forecasts
        # self.capacity_factor_profile_forecast = None # variable load factor per each sim period
        # self.generation_profile_forecast = None #[i * self.kW_p for i in self.capacity_factor_profile_forecast]

    def get_actual_generation_profile(self):
        """
        Compute generation schedule for entire simulation
        :return:
        """
        # data source: https://www.nrel.gov/grid/solar-power-data.html
        # data source: https://data.open-power-system-data.org/ninja_pv_wind_profiles/

        generation_profile_actual = prep.get_sim_PV_load_factors(
            base_path=self.base_path,
            cache_path=self.cache_path,
            sim_start_day=self.sim_start_day,
            sim_duration=self.sim_duration,
            num_lookback_periods=self.num_lookback_periods,
        )
        # generation_profile_actual = [i * self.kW_peak for i in capacity_factor_profile_actual]
        generation_profile_actual["pv_generation"] = generation_profile_actual[
            "pv_load_factor"
        ].apply(lambda x: x * self.kW_peak)
        generation_profile_actual.drop(["pv_load_factor"], axis=1, inplace=True)

        # print(generation_profile_actual)

        return generation_profile_actual

    def get_forecast_generation_profile(self):
        """
        Compute generation forecast for entire simulation
        --> Currently this is a persistence model that predicts the value from 24h ago
        :return:
        """
        generation_profile_actual = self.get_actual_generation_profile()
        generation_profile_forecast = generation_profile_actual.copy()

        # shift back by num_lookback_periods
        generation_profile_forecast["sim_time_shifted"] = (
            generation_profile_forecast.index + self.num_lookback_periods
        )
        generation_profile_forecast.set_index(
            "sim_time_shifted", drop=True, inplace=True
        )

        # print(generation_profile_forecast)

        return generation_profile_forecast


class DispatchableGenerator(ElectricGenerator):

    def __init__(self):
        ElectricGenerator.__init__(self)  # parent class
        # ....add attributes


# def __init__(self,env,kW_p,power_factor_profile_forecast,power_factor_profile):
#    self.env = env  # simulation environment
#    self.kW_p = kW_p  # peak capacity of all solar modules
#    # predicted values to possible model uncertainty
#    self.capacity_factor_profile_forecast = power_factor_profile_forecast # variable load factor per each sim period
#    self.generation_profile_forecast = power_factor_profile_forecast * kW_p  # variable generation profile over simulation horizon
#    # actual values
#    self.power_factor_profile = power_factor_profile  # variable load factor per each sim period
#    self.generation_profile = power_factor_profile * kW_p  # variable generation profile over simulation horizon
