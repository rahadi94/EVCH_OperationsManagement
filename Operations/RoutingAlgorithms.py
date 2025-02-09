from Environment.log import lg
import random


def random_charger_assignment(
    charging_stations,
    number_of_connectors,
    request,
    demand_threshold=1,
    duration_threshold=100 * 60,
):
    """
    Charger is assigned at random. This is equivalent to no active routing management
    (as would be sufficient in cases without parallel use of infrastructure)
    :param number_of_connectors:
    :param charging_stations:
    :return:
    """
    available_CSs = [
        x for x in charging_stations if x.connectors.count < number_of_connectors
    ]

    # ensure only EVs that meet criteria are allocated a charger
    if (
        request.ev == 1
        and request.energy_requested >= demand_threshold
        and request.park_duration <= duration_threshold
    ):

        if len(available_CSs) > 0:  # ensure chargers are available!
            charging_station = random.choice(
                available_CSs
            )  # picks randomly from available
        else:
            charging_station = None
            lg.info(
                f"Request {request.id} cannot be served (no available charging station)"
            )

    else:
        charging_station = None
    return charging_station


def lowest_occupancy_first_charger_assignment(
    charging_stations,
    number_of_connectors,
    request,
    demand_threshold=1,
    duration_threshold=100 * 60,
):
    """
    Lowest Occupancy Highest Priority (LOHP) charger assignment algorithm. This algorithm is also referred to
    as Worst Fit Routing in the bin packing literature (i.e., item is packed into the bin with the largest free capacity)
    :param number_of_connectors:
    :param charging_stations:
    :return:
    """
    available_CSs = [
        x for x in charging_stations if x.connectors.count < number_of_connectors
    ]
    # ensure only EVs that meet criteria are allocated a charger
    if (
        request.ev == 1
        and request.energy_requested >= demand_threshold
        and request.park_duration <= duration_threshold
    ):

        if len(available_CSs) > 0:  # ensure chargers are available!
            occupancy = [len(x.connected_vehicles) for x in available_CSs]
            lowest_occupancy = min(occupancy)
            charging_station = [
                x for x in available_CSs if x.connectors.count == lowest_occupancy
            ][
                0
            ]  # TODO: ENSURE YOU PICK IN ORDER OF ASCENDING ID
        else:
            charging_station = None
            lg.info(
                f"Request {request.id} cannot be served (no available charging station)"
            )

    else:
        charging_station = None
    return charging_station


def fill_one_after_other_charger_assignment(
    charging_stations,
    number_of_connectors,
    request,
    demand_threshold=1,
    duration_threshold=24 * 60,
):
    """
    Fill One After The Other (FOAO) charger assignment algorithm. This algorithm is also referred to
    as First Fit Routing in the bin packing literature (i.e., chargers are fully filled one after the other (in order of ID)).
    This algo assumes that users
    :param number_of_connectors:
    :param charging_stations:
    :return:
    """
    available_CSs = [
        x for x in charging_stations if x.connectors.count < number_of_connectors
    ]
    # ensure only EVs that meet criteria are allocated a charger
    if (
        request.ev == 1
        and request.energy_requested >= demand_threshold
        and request.park_duration <= duration_threshold
    ):

        if len(available_CSs) > 0:  # ensure chargers are available!
            occupancy = [len(x.connected_vehicles) for x in available_CSs]
            ids = [len(x.id) for x in available_CSs]
            lowest_occupancy = min(occupancy)
            smallest_id = min(ids)
            charging_stations = [
                x
                for x in available_CSs
                if x.connectors.count == lowest_occupancy and x.id == smallest_id
            ]
            # sort by ascending ID (lowest first)
            charging_stations.sort(key=lambda x: x.id)
            charging_station = charging_stations[0]
    else:
        charging_station = None
        lg.info(
            f"Request {request.id} cannot be served (no available charging station)"
        )

    return charging_station


def lowest_utilization_first_charger_assignment(
    charging_stations,
    number_of_connectors,
    request,
    demand_threshold=1,
    duration_threshold=24 * 60,
):
    """
    Lowest Utilization Highest Priority (LUHP) charger assignment algorithm.
    :param number_of_connectors:
    :param charging_stations:
    :return:
    """
    available_CSs = [
        x for x in charging_stations if x.connectors.count < number_of_connectors
    ]

    # ensure only EVs that meet criteria are allocated a charger
    if (
        request.ev == 1
        and request.energy_requested >= demand_threshold
        and request.park_duration <= duration_threshold
    ):
        if len(available_CSs) > 0:  # ensure chargers are available!
            power_usages = [x.current_power_usage for x in available_CSs]
            lowest_power_usage = min(power_usages)
            charging_stations = [
                x for x in available_CSs if x.current_power_usage == lowest_power_usage
            ]
            # sort by ascending ID (lowest first)
            charging_stations.sort(key=lambda x: x.id)
            charging_station = charging_stations[0]
        else:
            charging_station = None
            lg.info(
                f"Request {request.id} cannot be served (no available charging station)"
            )
    else:
        charging_station = None
    return charging_station


def matching_supply_demand_level(charging_stations, request):
    available_CSs = [
        x for x in charging_stations if x.connectors.count < x.number_of_connectors
    ]
    if request.ev == 1:
        if len(available_CSs) > 0:
            proper_CSs = [
                x
                for x in available_CSs
                if x.free_capacity_level == request.average_power_requirement_level
            ]
            if len(proper_CSs) > 0:
                return proper_CSs[0]  # TODO: maybe later we should distinguish them too
            else:
                proper_CSs = [
                    x
                    for x in available_CSs
                    if x.free_capacity_level > request.average_power_requirement_level
                ]
                if len(proper_CSs) > 0:
                    return proper_CSs[0]
                else:
                    proper_CSs = [
                        x
                        for x in available_CSs
                        if x.free_capacity_level
                        > request.average_power_requirement_level
                    ]
                    return proper_CSs[0]
        else:
            lg.info(
                f"Request {request.id} cannot be served (no available charging station)"
            )
            return None
    else:
        return None


def assign_to_the_minimum_power(charging_stations, request, demand_threshold=1):
    available_CSs = [
        x for x in charging_stations if x.connectors.count < x.number_of_connectors
    ]
    if request.ev == 1 and request.energy_requested >= demand_threshold:
        if len(available_CSs) > 0:
            proper_CSs = [
                x
                for x in available_CSs
                if x.free_capacity >= request.average_power_requirement * 1 + 5
            ]
            if len(proper_CSs) > 0:
                return sorted(proper_CSs, key=lambda x: x.free_capacity)[0]
            else:
                return sorted(available_CSs, key=lambda x: x.free_capacity)[-1]
        else:
            available_CSs = [x for x in charging_stations]
            min_free_time_list = [x.min_free_time for x in available_CSs]
            request.estimated_waiting_time = sorted(min_free_time_list)[0]
            if (
                sorted(min_free_time_list)[0] > request.departure_period
                or (sorted(min_free_time_list)[0] - request.arrival_period) > 10
            ):
                return None
            return sorted(available_CSs, key=lambda x: x.min_free_time)[0]
            # lg.info(f'Request {request.id} cannot be served (no available charging station)')
            # return None
    else:
        return None
