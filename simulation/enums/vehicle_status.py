"""
Vehicle status enumeration for type-safe vehicle state management.

This module defines the possible states a vehicle can be in during the simulation,
providing type safety and preventing string-based errors.
"""

from enum import Enum, auto
from typing import Set


class VehicleStatus(Enum):
    """
    Enumeration of possible vehicle states in the simulation.
    
    The states follow a logical progression through the vehicle lifecycle:
    1. NOT_ARRIVED -> Vehicle not yet arrived (initial state)
    2. ARRIVED -> Vehicle arrives at facility
    3. ASSIGNED -> Vehicle assigned to charger/parking
    4. CONNECTED -> Vehicle connected to charger (ready to charge)
    5. CHARGING -> Vehicle actively charging
    6. FULLY_CHARGED -> Vehicle finished charging
    7. LEFT -> Vehicle left the facility
    
    For non-EV vehicles:
    1. NOT_ARRIVED -> Vehicle not yet arrived (initial state)
    2. ARRIVED -> Vehicle arrives at facility
    3. PARKING -> Vehicle is parking
    4. LEFT -> Vehicle left the facility
    """
    
    # Initial state
    NOTARRIVED = "notarrived"
    ARRIVED = "arrived"
    
    # Assignment states
    ASSIGNED = "assigned"
    CONNECTED = "connected"
    
    # Charging states
    CHARGING = "charging"
    FULLY_CHARGED = "fully_charged"
    
    # Parking state (for non-EV vehicles)
    PARKING = "parking"
    
    # Final state
    LEFT = "left"
    
    # Legacy states (for backward compatibility during transition)
    FINISHED = "finished"  # Alias for FULLY_CHARGED
    
    @classmethod
    def get_charging_states(cls) -> Set['VehicleStatus']:
        """Get all states where vehicle can receive charging power."""
        return {cls.ASSIGNED, cls.CONNECTED, cls.CHARGING}
    
    @classmethod
    def get_active_states(cls) -> Set['VehicleStatus']:
        """Get all states where vehicle is present at facility."""
        return {
            cls.ARRIVED, cls.ASSIGNED, cls.CONNECTED, 
            cls.CHARGING, cls.FULLY_CHARGED, cls.PARKING
        }
    
    @classmethod
    def get_ev_states(cls) -> Set['VehicleStatus']:
        """Get all states relevant for EV vehicles."""
        return {
            cls.NOTARRIVED, cls.ARRIVED, cls.ASSIGNED, cls.CONNECTED,
            cls.CHARGING, cls.FULLY_CHARGED, cls.LEFT
        }
    
    @classmethod
    def get_non_ev_states(cls) -> Set['VehicleStatus']:
        """Get all states relevant for non-EV vehicles."""
        return {cls.NOTARRIVED, cls.ARRIVED, cls.PARKING, cls.LEFT}
    
    def is_charging_eligible(self) -> bool:
        """Check if vehicle can receive charging power."""
        return self in self.get_charging_states()
    
    def is_active(self) -> bool:
        """Check if vehicle is still at the facility."""
        return self in self.get_active_states()
    
    def is_ev_state(self) -> bool:
        """Check if this is a valid state for EV vehicles."""
        return self in self.get_ev_states()
    
    def is_non_ev_state(self) -> bool:
        """Check if this is a valid state for non-EV vehicles."""
        return self in self.get_non_ev_states()
    
    def __str__(self) -> str:
        """Return string representation for backward compatibility."""
        return self.value


class ChargerStatus(Enum):
    """
    Enumeration of possible charger states.
    """
    IDLE = "idle"
    CONNECTING = "connecting"
    CHARGING = "charging"
    DISCONNECTING = "disconnecting"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    
    def __str__(self) -> str:
        """Return string representation for backward compatibility."""
        return self.value


class StorageStatus(Enum):
    """
    Enumeration of possible energy storage states.
    """
    IDLE = "idle"
    CHARGING = "charging"
    DISCHARGING = "discharging"
    ERROR = "error"
    MAINTENANCE = "maintenance"
    
    def __str__(self) -> str:
        """Return string representation for backward compatibility."""
        return self.value
