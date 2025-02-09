import simpy
from simpy import Resource
from Environment.log import lg


class ParkingLot:
    """
    The parking lot is modelled as a simple resource which all vehicles request upon entry.
    For the simulation it does not matter where a particular parking spot is located within the EVCC.
    """

    def __init__(self, env, parking_capacity):
        self.env = env
        self.info = {}
        self.parking_spots = simpy.Resource(self.env, capacity=parking_capacity)
        self.parked_vehicles = []  # not really needed
        self.occupancy = []

    def park(self, vehicle):
        """
        Vehicle enters parking lot and requests 1 unit of parking resource
        :param vehicle:
        :return:
        """
        vehicle.assigned_parking = (
            self.parking_spots.request()
        )  # request a (i.e., any) parking spot
        yield vehicle.assigned_parking  # yield the request

        self.parked_vehicles.append(vehicle)
        vehicle.mode = "parking"
        lg.info(f"Vehicle {vehicle.id} parks at {self.env.now}")

    def leave(self, vehicle):
        """
        Vehicle leaves parking lot and releases 1 unit of parking resource
        :param vehicle:
        :return:
        """
        self.parking_spots.release(vehicle.assigned_parking)  # release the parking spot

        self.parked_vehicles.remove(vehicle)
        vehicle.mode = "leaving"
        lg.info(f"Vehicle {vehicle.id} leaves at {self.env.now}")

    def monitor(self):
        """
        Monitoring the status of parking resource every time step
        """
        self.info = dict(State=[])
        while True:
            self.info["State"].append(
                [self.parking_spots.count, len(self.parking_spots.queue)]
            )
            yield self.env.timeout(1)
