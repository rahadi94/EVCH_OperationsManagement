"""
Configuration facade for clean access to configuration values.

This module provides a clean interface to the configuration system,
reducing attribute clutter in model and operator classes.
"""

from resources.configuration.configuration import Configuration
from typing import Any, Dict, List, Optional
from dataclasses import dataclass


@dataclass
class PricingConfig:
    """Pricing configuration data."""
    energy_prices: List[float]
    max_price_tou: float
    default_charging_price: float
    default_charging_power: float
    peak_cost: float
    peak_threshold: float
    parking_price: float
    price_parameters: List[float]
    pricing_mode: str
    dynamic_pricing: bool
    dynamic_fix_term_pricing: bool


@dataclass
class ThresholdConfig:
    """Threshold configuration data."""
    demand_threshold: float
    duration_threshold: int
    peak_threshold: float
    minimum_served_demand: float


@dataclass
class SystemConfig:
    """System configuration data."""
    random_demand: bool
    data_source: str
    benchmarking: bool
    b2g: bool
    multiple_power: bool
    request_adjusting_mode: str
    dynamic_storage_scheduling: bool


class ConfigFacade:
    """
    Facade for accessing configuration values with caching.
    
    This class provides a clean interface to configuration values,
    reducing the need for individual configuration attributes in
    model and operator classes.
    """
    
    def __init__(self):
        self._config = Configuration.instance()
        self._cache: Dict[str, Any] = {}
    
    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value with caching.
        
        Args:
            key: Configuration key
            default: Default value if key not found
            
        Returns:
            Configuration value
        """
        if key not in self._cache:
            self._cache[key] = getattr(self._config, key, default)
        return self._cache[key]
    
    def clear_cache(self) -> None:
        """Clear the configuration cache."""
        self._cache.clear()
    
    # ============================================================================
    # PRICING CONFIGURATION
    # ============================================================================
    
    @property
    def pricing_config(self) -> PricingConfig:
        """Get pricing configuration as a structured object."""
        return PricingConfig(
            energy_prices=self.get('energy_prices', []),
            max_price_tou=self.get('max_price_ToU', 0.0),
            default_charging_price=self.get('default_charging_price', 0.0),
            default_charging_power=self.get('default_charging_power', 0.0),
            peak_cost=self.get('peak_cost', 0.0),
            peak_threshold=self.get('peak_threshold', 0.0),
            parking_price=self.get('parking_price', 0.0),
            price_parameters=self.get('price_parameters', []),
            pricing_mode=self.get('pricing_mode', 'static'),
            dynamic_pricing=self.get('dynamic_pricing', False),
            dynamic_fix_term_pricing=self.get('dynamic_fix_term_pricing', False)
        )
    
    @property
    def energy_prices(self) -> List[float]:
        """Get energy prices configuration."""
        return self.get('energy_prices', [])
    
    @property
    def max_price_tou(self) -> float:
        """Get maximum Time-of-Use price."""
        return self.get('max_price_ToU', 0.0)
    
    @property
    def peak_cost(self) -> float:
        """Get peak cost configuration."""
        return self.get('peak_cost', 0.0)
    
    @property
    def peak_threshold(self) -> float:
        """Get peak threshold configuration."""
        return self.get('peak_threshold', 0.0)
    
    @property
    def parking_price(self) -> float:
        """Get parking price configuration."""
        return self.get('parking_price', 0.0)
    
    @property
    def price_parameters(self) -> List[float]:
        """Get price parameters configuration."""
        return self.get('price_parameters', [])
    
    @property
    def pricing_mode(self) -> str:
        """Get pricing mode configuration."""
        return self.get('pricing_mode', 'static')
    
    @property
    def dynamic_pricing(self) -> bool:
        """Get dynamic pricing configuration."""
        return self.get('dynamic_pricing', False)
    
    @property
    def dynamic_fix_term_pricing(self) -> bool:
        """Get dynamic fix term pricing configuration."""
        return self.get('dynamic_fix_term_pricing', False)
    
    # ============================================================================
    # THRESHOLD CONFIGURATION
    # ============================================================================
    
    @property
    def threshold_config(self) -> ThresholdConfig:
        """Get threshold configuration as a structured object."""
        return ThresholdConfig(
            demand_threshold=self.get('demand_threshold', 0.0),
            duration_threshold=self.get('duration_threshold', 0),
            peak_threshold=self.get('peak_threshold', 0.0),
            minimum_served_demand=self.get('minimum_served_demand', 0.0)
        )
    
    @property
    def demand_threshold(self) -> float:
        """Get demand threshold configuration."""
        return self.get('demand_threshold', 0.0)
    
    @property
    def duration_threshold(self) -> int:
        """Get duration threshold configuration."""
        return self.get('duration_threshold', 0)
    
    # ============================================================================
    # SYSTEM CONFIGURATION
    # ============================================================================
    
    @property
    def system_config(self) -> SystemConfig:
        """Get system configuration as a structured object."""
        return SystemConfig(
            random_demand=self.get('random_demand', False),
            data_source=self.get('data_source', ''),
            benchmarking=self.get('benchmarking', False),
            b2g=self.get('B2G', False),
            multiple_power=self.get('multiple_power', False),
            request_adjusting_mode=self.get('request_adjusting_mode', ''),
            dynamic_storage_scheduling=self.get('dynamic_storage_scheduling', False)
        )
    
    @property
    def random_demand(self) -> bool:
        """Get random demand configuration."""
        return self.get('random_demand', False)
    
    @property
    def data_source(self) -> str:
        """Get data source configuration."""
        return self.get('data_source', '')
    
    @property
    def benchmarking(self) -> bool:
        """Get benchmarking configuration."""
        return self.get('benchmarking', False)
    
    @property
    def b2g(self) -> bool:
        """Get B2G (Battery-to-Grid) configuration."""
        return self.get('B2G', False)
    
    @property
    def multiple_power(self) -> bool:
        """Get multiple power configuration."""
        return self.get('multiple_power', False)
    
    @property
    def request_adjusting_mode(self) -> str:
        """Get request adjusting mode configuration."""
        return self.get('request_adjusting_mode', '')
    
    @property
    def dynamic_storage_scheduling(self) -> bool:
        """Get dynamic storage scheduling configuration."""
        return self.get('dynamic_storage_scheduling', False)
    
    # ============================================================================
    # UTILITY METHODS
    # ============================================================================
    
    def get_utility_constant(self) -> float:
        """Get utility constant configuration."""
        return self.get('utility_constant', 0.0)
    
    def get_utility_beta(self, hour: int, avg_power: float) -> float:
        """Get utility beta configuration."""
        if hasattr(self._config, 'get_utility_beta'):
            return self._config.get_utility_beta(hour, avg_power)
        return 0.0
    
    def get_utility_beta_parking(self, hour: int) -> float:
        """Get utility beta parking configuration."""
        if hasattr(self._config, 'get_utility_beta_parking'):
            return self._config.get_utility_beta_parking(hour)
        return 0.0
    
    def get_base_price(self, hour: int, avg_power: float) -> float:
        """Get base price configuration."""
        if hasattr(self._config, 'get_base_price'):
            return self._config.get_base_price(hour, avg_power)
        return 0.0
    
    def get_base_parking_fee(self, hour: int, avg_power: float) -> float:
        """Get base parking fee configuration."""
        if hasattr(self._config, 'get_base_parking_fee'):
            return self._config.get_base_parking_fee(hour, avg_power)
        return 0.0
    
    def get_base_power(self) -> float:
        """Get base power configuration."""
        if hasattr(self._config, 'get_base_power'):
            return self._config.get_base_power()
        return 0.0
