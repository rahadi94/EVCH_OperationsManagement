import simpy
from simpy import Container

from Environment.log import lg


class EVCharger:
    """
    Charger object with container for max_energy_period and resource of available_connectors
    """

    def __init__(self, env, id, power, period_length, number_of_connectors):
        """
        :param env:
        :param id:
        :param power: rated power of charger
        :param period_length: lenght of planning period in minutes (determines max. energy transfer per period)
        :param number_of_connectors:
        """
        self.min_free_time = None
        self.departure_times = []
        self.env = env
        self.id = id
        self.power = power  # kW
        self.available_power = power  # kW it might be less than the original power due to the grid restriction
        self.period_length = period_length  # periods
        # self.max_energy_period = self.power * (self.period_length/60) #kWh
        self.number_of_connectors = number_of_connectors
        # self.req_energy_period = simpy.Container(self.env,capacity=max_energy_transfer_period,init=max_energy_transfer_period) #kWh
        self.connectors = simpy.Resource(self.env, capacity=number_of_connectors)
        self.connected_vehicles = []
        self.in_queue_vehicles = []
        self.charging_vehicles = []
        self.current_power_usage = sum(
            [x.charging_power for x in self.connected_vehicles]
        )
        self.free_capacity = power
        self.free_capacity_level = 3
        self.energy_transfer = []
        self.info = {}  # information for final results

    def connect(self, vehicle):
        """
        Consume a plug of the charging station and assign request to vehicle
        :param vehicle:
        :return:
        """
        vehicle.assigned_charger = self.connectors.request()
        yield vehicle.assigned_charger
        self.connected_vehicles.append(vehicle)
        vehicle.mode = "connected"
        lg.info(
            f"Vehicle {vehicle.id} connects to charging_station {self.id} at {self.env.now}"
        )

    def charge(self, vehicle, energy):
        """
        Charge vehicle and consume energy
        :param vehicle:
        :param energy:
        :return:
        """
        # we adopt charging rate in a discrete time setting, i.e. energy is transferred over 1 planning period

        vehicle.assigned_energy = self.req_energy_period.get(
            energy
        )  # request energy from the charger
        yield vehicle.assigned_energy  # yield request
        vehicle.energy_charged += energy
        yield (self.env.timeout(self.period_length))  # delivering energy takes 1 period
        self.req_energy_period.put(
            energy
        )  # release charge energy at end of period (will be re-assigned in new period!)

        vehicle.mode = "Connected"
        lg.info(
            f"Vehicle {vehicle.id} charges at charging_station {self.id} at {self.env.now}"
        )

    def disconnect(self, vehicle):
        """
        Release connector
        :param vehicle:
        :return:
        """
        self.connectors.release(vehicle.assigned_charger)  # release the connector
        vehicle.assigned_charger = None  # remove charger request from vehicle class

        vehicle.mode = "finished"
        lg.info(
            f"Vehicle {vehicle.id} finishes charging at {self.env.now} with total {vehicle.energy_charged} kWh delivered"
        )

    def get_free_capacity_level(self):
        if self.free_capacity >= 40:
            return 3
        elif self.free_capacity >= 20:
            return 2
        else:
            return 1

    def status_update(self):
        self.current_power_usage = 0
        for vehicle in [x for x in self.connected_vehicles if x.mode == "Connected"]:
            self.current_power_usage += vehicle.charging_power
        self.charging_vehicles = [
            x for x in self.connected_vehicles if x.mode == "Connected"
        ]
        self.free_capacity = self.power - self.current_power_usage
        self.free_capacity_level = self.get_free_capacity_level()
        self.departure_times = [x.departure_period for x in self.connected_vehicles] + [
            x.departure_period for x in self.in_queue_vehicles
        ]
        # if len(self.in_queue_vehicles)>0:
        #     print(self.departure_times, len(self.connectors.queue))
        if len(self.departure_times) > 0:
            self.min_free_time = sorted(self.departure_times)[
                len(self.connectors.queue)
            ]

    ##############################
    # MONITORING

    def monitor(self):
        """
        Monitoring the status of chargers (number of connected and charging vehicles) every time step
        """
        self.info["Connected"] = []
        self.info["Charging"] = []
        self.info["Consumption"] = []
        while True:
            charging_vehicles = [
                x for x in self.connected_vehicles if x.mode == "Connected"
            ]
            Num_charging = len(charging_vehicles)
            Num_connected = len(self.connected_vehicles)
            consumption = 0
            for vehicle in charging_vehicles:
                consumption += vehicle.charging_power / 60
            self.info["Connected"].append(Num_connected)
            self.info["Charging"].append(Num_charging)
            self.info["Consumption"].append(consumption)

            yield self.env.timeout(1)
