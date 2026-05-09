"""
Traffic Congestion Estimation Module

This module provides functionality to estimate traffic congestion using
Volume-to-Capacity (V/C) ratio and Level of Service (LOS) metrics.

Supports two modes:
1. Time-based: Uses time intervals and historical vehicle counts
2. Real-time: Uses current frame vehicle count vs. fixed capacity
"""

from collections import deque
from typing import Dict, Tuple, Optional
import time


class RealtimeTrafficCongestionEstimator:
    """
    Real-time traffic congestion estimator based on current frame vehicle count.
    
    This estimator calculates:
    - Volume: Number of vehicles currently visible in frame (weighted by PCE)
    - Capacity: User-defined maximum vehicle capacity
    - V/C ratio: Real-time volume/capacity ratio
    - Level of Service (LOS) based on V/C ratio
    """
    
    # Vehicle category multipliers (Passenger Car Equivalent - PCE)
    VEHICLE_MULTIPLIERS = {
        'Car': 1.0,
        'Motorcycle': 1.0,
        'Tricycle': 2.5,
        'Bus': 2.0,
        'Van': 1.5,
        'Truck': 2.5
    }
    
    # LOS thresholds based on V/C ratio
    LOS_THRESHOLDS = [
        ('A', 0.0, 0.20),    # 0.0 <= x <= 0.20
        ('B', 0.20, 0.50),   # 0.20 < x <= 0.50
        ('C', 0.50, 0.70),   # 0.50 < x <= 0.70
        ('D', 0.70, 0.85),   # 0.70 < x <= 0.85
        ('E', 0.85, 1.00),   # 0.85 < x <= 1.00
        ('F', 1.00, float('inf'))  # x > 1.00
    ]
    
    JAM_DENSITY = 145  # vehicles/km/lane

    def __init__(self, road_length_km: float, num_lanes: int, jam_density: float = JAM_DENSITY):
        """
        Initialize the real-time traffic congestion estimator.
        
        Args:
            road_length_km: Length of roadway in kilometers
            num_lanes: Number of lanes
            jam_density: Jam density (vehicles/km/lane)
        """
        self.road_length_km = road_length_km
        self.num_lanes = num_lanes
        self.jam_density = jam_density
        self.capacity = self._calculate_capacity()

    def _calculate_capacity(self) -> float:
        """
        Calculate maximum vehicle capacity using jam density.

        N_max = N_lanes * L * k_j
        """
        return self.num_lanes * self.road_length_km * self.jam_density
    
    def calculate_volume(self, vehicle_counts: Dict[str, int]) -> float:
        """
        Calculate traffic volume based on current vehicle counts in frame.
        
        Volume is calculated as the sum of (vehicle_count * multiplier) for each
        vehicle category currently visible.
        
        Args:
            vehicle_counts: Dictionary mapping class names to counts
            
        Returns:
            Traffic volume in Passenger Car Equivalent (PCE) units
        """
        volume = 0.0
        for class_name, count in vehicle_counts.items():
            multiplier = self.VEHICLE_MULTIPLIERS.get(class_name, 1.0)
            volume += count * multiplier
        
        return volume
    
    def calculate_vc_ratio(self, vehicle_counts: Dict[str, int]) -> float:
        """
        Calculate Volume-to-Capacity (V/C) ratio.
        
        V/C ratio = Volume / Capacity
        
        Args:
            vehicle_counts: Dictionary mapping class names to counts
            
        Returns:
            V/C ratio
        """
        volume = self.calculate_volume(vehicle_counts)
        
        if self.capacity == 0:
            return 0.0
        
        vc_ratio = volume / self.capacity
        return vc_ratio
    
    def get_los(self, vc_ratio: float) -> str:
        """
        Get Level of Service (LOS) based on V/C ratio.
        
        LOS Categories:
        - A: Free flow (x < 0.20)
        - B: Reasonably free flow (0.21 <= x <= 0.50)
        - C: Stable flow (0.51 <= x <= 0.70)
        - D: Approaching unstable flow (0.71 <= x <= 0.85)
        - E: Unstable flow (0.86 <= x <= 1.00)
        - F: Forced or breakdown flow (x > 1.00)
        
        Args:
            vc_ratio: V/C ratio
            
        Returns:
            LOS category ('A', 'B', 'C', 'D', 'E', or 'F')
        """
        for los, min_val, max_val in self.LOS_THRESHOLDS:
            # For 'A', include lower bound; for others, use strict lower bound
            if los == 'A':
                if min_val <= vc_ratio <= max_val:
                    return los
            else:
                if min_val < vc_ratio <= max_val:
                    return los
        # Fallback (should not happen)
        return 'F'
    
    def get_congestion_status(self, vehicle_counts: Dict[str, int]) -> Dict[str, any]:
        """
        Get comprehensive congestion status including volume, V/C ratio, and LOS.
        
        Args:
            vehicle_counts: Dictionary mapping class names to counts
            
        Returns:
            Dictionary containing:
            - volume: Traffic volume in PCE
            - capacity: Maximum capacity
            - vc_ratio: Volume-to-Capacity ratio
            - los: Level of Service
            - vehicle_counts: Dictionary of vehicle counts by class
        """
        volume = self.calculate_volume(vehicle_counts)
        vc_ratio = self.calculate_vc_ratio(vehicle_counts)
        los = self.get_los(vc_ratio)
        
        return {
            'volume': volume,
            'capacity': self.capacity,
            'vc_ratio': vc_ratio,
            'los': los,
            'vehicle_counts': dict(vehicle_counts)
        }
    
    @staticmethod
    def get_los_description(los: str) -> str:
        """
        Get a human-readable description of the LOS level.
        
        Args:
            los: LOS category ('A' through 'F')
            
        Returns:
            Description of the traffic condition
        """
        descriptions = {
            'A': 'Free flow - Excellent conditions',
            'B': 'Reasonably free flow - Good conditions',
            'C': 'Stable flow - Acceptable conditions',
            'D': 'Approaching unstable - Moderate congestion',
            'E': 'Unstable flow - Heavy congestion',
            'F': 'Forced/breakdown flow - Severe congestion'
        }
        return descriptions.get(los, 'Unknown')
    
    @staticmethod
    def get_los_color(los: str) -> Tuple[int, int, int]:
        """
        Get BGR color for visualizing LOS level.
        
        Args:
            los: LOS category ('A' through 'F')
            
        Returns:
            Tuple of (B, G, R) values for OpenCV
        """
        colors = {
            'A': (0, 255, 0),      # Green - Excellent
            'B': (0, 200, 100),    # Light green - Good
            'C': (0, 255, 255),    # Yellow - Acceptable
            'D': (0, 165, 255),    # Orange - Moderate
            'E': (0, 100, 255),    # Dark orange - Heavy
            'F': (0, 0, 255)       # Red - Severe
        }
        return colors.get(los, (128, 128, 128))  # Gray for unknown


class TrafficCongestionEstimator:
    """
    Estimates traffic congestion based on vehicle counts and road characteristics.
    
    The estimator calculates:
    - Volume-to-Capacity (V/C) ratio
    - Level of Service (LOS) based on V/C ratio
    """
    
    # Vehicle category multipliers (Passenger Car Equivalent - PCE)
    VEHICLE_MULTIPLIERS = {
        'Car': 1.0,
        'Motorcycle': 1.0,
        'Tricycle': 2.5,
        'Bus': 2.0,
        'Van': 1.5,
        'Truck': 2.5
    }
    
    # LOS thresholds based on V/C ratio
    LOS_THRESHOLDS = [
        ('A', 0.0, 0.20),   # x < 0.20
        ('B', 0.21, 0.50),  # 0.21 <= x <= 0.50
        ('C', 0.51, 0.70),  # 0.51 <= x <= 0.70
        ('D', 0.71, 0.85),  # 0.71 <= x <= 0.85
        ('E', 0.86, 1.00),  # 0.86 <= x <= 1.00
        ('F', 1.01, float('inf'))  # x > 1.00
    ]
    
    def __init__(self, road_width: float, time_interval: float, num_lanes: int = 1):
        """
        Initialize the traffic congestion estimator.
        
        Args:
            road_width: Width of the road in meters
            time_interval: Time interval for volume calculation in minutes
            num_lanes: Number of lanes (default: 1)
        """
        self.road_width = road_width
        self.time_interval = time_interval
        self.num_lanes = num_lanes
        self.capacity = self._calculate_capacity(road_width, num_lanes)
        
        # Store vehicle counts with timestamps
        # Format: deque of (timestamp, class_name, count) tuples
        self.vehicle_history = deque()
        
        # Current vehicle counts by class
        self.current_counts = {}
        
    def _calculate_capacity(self, road_width: float, num_lanes: int = 1) -> int:
        """
        Calculate hourly road capacity based on road width and number of lanes.
        
        Args:
            road_width: Width of road in meters
            num_lanes: Number of lanes
            
        Returns:
            Hourly capacity in vehicles per hour
        """
        # Calculate capacity per lane based on road width
        if road_width < 4.0:
            capacity_per_lane = 600
        elif 4.0 <= road_width <= 5.0:
            capacity_per_lane = 1200
        elif 5.1 <= road_width <= 6.0:
            capacity_per_lane = 1600
        elif 6.1 <= road_width <= 6.7:
            capacity_per_lane = 1700
        elif road_width >= 6.8:
            capacity_per_lane = 1800
        else:
            # Default fallback
            capacity_per_lane = 600
        
        # Multiply by number of lanes to get total capacity
        return capacity_per_lane * num_lanes
    
    def add_vehicle(self, class_name: str, timestamp: Optional[float] = None):
        """
        Add a vehicle to the counting system.
        
        Args:
            class_name: Name of the vehicle class (e.g., 'Car', 'Bus', etc.)
            timestamp: Unix timestamp when vehicle was counted (defaults to current time)
        """
        if timestamp is None:
            timestamp = time.time()
        
        self.vehicle_history.append((timestamp, class_name))
        
        # Update current counts
        if class_name not in self.current_counts:
            self.current_counts[class_name] = 0
        self.current_counts[class_name] += 1
    
    def _cleanup_old_records(self, current_time: float):
        """
        Remove vehicle records older than the time interval.
        
        Args:
            current_time: Current timestamp
        """
        time_threshold = current_time - (self.time_interval * 60)  # Convert minutes to seconds
        
        # Remove old entries from the front of the deque
        while self.vehicle_history and self.vehicle_history[0][0] < time_threshold:
            old_timestamp, old_class = self.vehicle_history.popleft()
            # Decrement count (safety check)
            if old_class in self.current_counts and self.current_counts[old_class] > 0:
                self.current_counts[old_class] -= 1
    
    def calculate_volume(self, current_time: Optional[float] = None) -> float:
        """
        Calculate traffic volume based on vehicle counts within the time interval.
        
        Volume is calculated as the sum of (vehicle_count * multiplier) for each
        vehicle category within the specified time interval.
        
        Args:
            current_time: Current timestamp (defaults to current time)
            
        Returns:
            Traffic volume in Passenger Car Equivalent (PCE) units
        """
        if current_time is None:
            current_time = time.time()
        
        # Clean up old records
        self._cleanup_old_records(current_time)
        
        # Calculate volume using PCE multipliers
        volume = 0.0
        for class_name, count in self.current_counts.items():
            multiplier = self.VEHICLE_MULTIPLIERS.get(class_name, 1.0)
            volume += count * multiplier
        
        return volume
    
    def calculate_vc_ratio(self, current_time: Optional[float] = None) -> float:
        """
        Calculate Volume-to-Capacity (V/C) ratio.
        
        V/C ratio = Volume / (Capacity * (time_interval / 60))
        
        The capacity is adjusted based on the time interval since capacity
        is given in vehicles per hour.
        
        Args:
            current_time: Current timestamp (defaults to current time)
            
        Returns:
            V/C ratio
        """
        volume = self.calculate_volume(current_time)
        
        # Adjust capacity for the time interval
        # Capacity is hourly, so we scale it down for shorter intervals
        interval_hours = self.time_interval / 60.0
        adjusted_capacity = self.capacity * interval_hours
        
        if adjusted_capacity == 0:
            return 0.0
        
        vc_ratio = volume / adjusted_capacity
        return vc_ratio
    
    def get_los(self, vc_ratio: Optional[float] = None, current_time: Optional[float] = None) -> str:
        """
        Get Level of Service (LOS) based on V/C ratio.
        
        LOS Categories:
        - A: Free flow (x < 0.20)
        - B: Reasonably free flow (0.21 <= x <= 0.50)
        - C: Stable flow (0.51 <= x <= 0.70)
        - D: Approaching unstable flow (0.71 <= x <= 0.85)
        - E: Unstable flow (0.86 <= x <= 1.00)
        - F: Forced or breakdown flow (x > 1.00)
        
        Args:
            vc_ratio: V/C ratio (if None, will be calculated)
            current_time: Current timestamp (used if vc_ratio is None)
            
        Returns:
            LOS category ('A', 'B', 'C', 'D', 'E', or 'F')
        """
        if vc_ratio is None:
            vc_ratio = self.calculate_vc_ratio(current_time)
        
        for los, min_val, max_val in self.LOS_THRESHOLDS:
            if min_val <= vc_ratio <= max_val:
                return los
        
        # Fallback (should not happen)
        return 'F'
    
    def get_congestion_status(self, current_time: Optional[float] = None) -> Dict[str, any]:
        """
        Get comprehensive congestion status including volume, V/C ratio, and LOS.
        
        Args:
            current_time: Current timestamp (defaults to current time)
            
        Returns:
            Dictionary containing:
            - volume: Traffic volume in PCE
            - capacity: Adjusted road capacity for the time interval
            - vc_ratio: Volume-to-Capacity ratio
            - los: Level of Service
            - vehicle_counts: Dictionary of vehicle counts by class
            - time_interval: Time interval in minutes
            - road_width: Road width in meters
        """
        if current_time is None:
            current_time = time.time()
        
        volume = self.calculate_volume(current_time)
        vc_ratio = self.calculate_vc_ratio(current_time)
        los = self.get_los(vc_ratio)
        
        interval_hours = self.time_interval / 60.0
        adjusted_capacity = self.capacity * interval_hours
        
        return {
            'volume': volume,
            'capacity': adjusted_capacity,
            'hourly_capacity': self.capacity,
            'vc_ratio': vc_ratio,
            'los': los,
            'vehicle_counts': dict(self.current_counts),
            'time_interval': self.time_interval,
            'road_width': self.road_width
        }
    
    def reset(self):
        """Reset all vehicle counts and history."""
        self.vehicle_history.clear()
        self.current_counts.clear()
    
    @staticmethod
    def get_los_description(los: str) -> str:
        """
        Get a human-readable description of the LOS level.
        
        Args:
            los: LOS category ('A' through 'F')
            
        Returns:
            Description of the traffic condition
        """
        descriptions = {
            'A': 'Free flow - Excellent conditions',
            'B': 'Reasonably free flow - Good conditions',
            'C': 'Stable flow - Acceptable conditions',
            'D': 'Approaching unstable - Moderate congestion',
            'E': 'Unstable flow - Heavy congestion',
            'F': 'Forced/breakdown flow - Severe congestion'
        }
        return descriptions.get(los, 'Unknown')
    
    @staticmethod
    def get_los_color(los: str) -> Tuple[int, int, int]:
        """
        Get BGR color for visualizing LOS level.
        
        Args:
            los: LOS category ('A' through 'F')
            
        Returns:
            Tuple of (B, G, R) values for OpenCV
        """
        colors = {
            'A': (0, 255, 0),      # Green - Excellent
            'B': (0, 200, 100),    # Light green - Good
            'C': (0, 255, 255),    # Yellow - Acceptable
            'D': (0, 165, 255),    # Orange - Moderate
            'E': (0, 100, 255),    # Dark orange - Heavy
            'F': (0, 0, 255)       # Red - Severe
        }
        return colors.get(los, (128, 128, 128))  # Gray for unknown
