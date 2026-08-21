"""
Configuration loader for the mineral pore segmentation pipeline.
Provides easy access to configuration parameters with validation.
"""

import yaml
import os
from typing import Dict, Any, Optional
from pathlib import Path


class ConfigLoader:
    """Load and validate pipeline configuration."""
    
    def __init__(self, config_path: str = "config/pipeline_config.yaml"):
        """
        Initialize configuration loader.
        
        Args:
            config_path: Path to YAML configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self._validate_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    def _validate_config(self):
        """Validate configuration parameters."""
        # Check required sections exist
        required_sections = ['paths', 'image_processing', 'yellow_detection', 
                           'gpu_processing', 'model', 'validation']
        
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required config section: {section}")
        
        # Validate image sizes
        orig_size = self.config['image_processing']['original_size']
        target_size = self.config['image_processing']['target_size']
        
        if not all(isinstance(s, list) and len(s) == 2 for s in [orig_size, target_size]):
            raise ValueError("Image sizes must be [width, height] lists")
        
        if any(t > o for t, o in zip(target_size, orig_size)):
            raise ValueError("Target size cannot be larger than original size")
    
    def get(self, key_path: str, default: Any = None) -> Any:
        """
        Get configuration value using dot notation.
        
        Args:
            key_path: Dot-separated path to config value (e.g., 'model.epochs')
            default: Default value if key not found
        
        Returns:
            Configuration value
        """
        keys = key_path.split('.')
        value = self.config
        
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        
        return value
    
    def update(self, key_path: str, value: Any):
        """
        Update configuration value at runtime.
        
        Args:
            key_path: Dot-separated path to config value
            value: New value to set
        """
        keys = key_path.split('.')
        config = self.config
        
        for key in keys[:-1]:
            if key not in config:
                config[key] = {}
            config = config[key]
        
        config[keys[-1]] = value
    
    def get_paths(self) -> Dict[str, Path]:
        """Get all paths as Path objects."""
        paths = {}
        for key, value in self.config['paths'].items():
            paths[key] = Path(value)
        return paths
    
    def create_directories(self):
        """Create all required directories."""
        paths = self.get_paths()
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
    
    def save(self, output_path: Optional[str] = None):
        """
        Save current configuration to file.
        
        Args:
            output_path: Path to save config (defaults to original path)
        """
        save_path = output_path or self.config_path
        
        with open(save_path, 'w') as f:
            yaml.dump(self.config, f, default_flow_style=False, sort_keys=False)
    
    def __repr__(self):
        return f"ConfigLoader(config_path='{self.config_path}')"


# Convenience function
def load_config(config_path: str = "config/pipeline_config.yaml") -> ConfigLoader:
    """Load configuration from file."""
    return ConfigLoader(config_path)