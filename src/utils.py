import os
import json
import logging
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional, Union, Tuple
import yaml

def ensure_dir(path: Union[str, Path]) -> str:
    """Create directory if it doesn't exist"""
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)

def get_project_root() -> Path:
    """Return project root directory"""
    return Path(__file__).parent.parent

def load_json(file_path: Union[str, Path]) -> Dict:
    """Load JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def save_json(data: Dict, file_path: Union[str, Path], indent: int = 2) -> None:
    """Save data to JSON file"""
    ensure_dir(Path(file_path).parent)
    with open(file_path, 'w') as f:
        json.dump(data, f, indent=indent)

def load_csv_data(file_path: Union[str, Path], **kwargs) -> pd.DataFrame:
    """Load CSV file into DataFrame"""
    return pd.read_csv(file_path, **kwargs)

def save_csv_data(data: pd.DataFrame, file_path: Union[str, Path], **kwargs) -> None:
    """Save DataFrame to CSV file"""
    ensure_dir(Path(file_path).parent)
    data.to_csv(file_path, **kwargs)

def load_pickle(file_path: Union[str, Path]) -> Any:
    """Load pickled object"""
    with open(file_path, 'rb') as f:
        return pickle.load(f)

def save_pickle(obj: Any, file_path: Union[str, Path]) -> None:
    """Save object as pickle file"""
    ensure_dir(Path(file_path).parent)
    with open(file_path, 'wb') as f:
        pickle.dump(obj, f)

def get_timestamp() -> str:
    """Get current timestamp as string"""
    return datetime.now().strftime('%Y%m%d_%H%M%S')

def setup_logging(config_file: Optional[str] = None, default_level: int = logging.INFO) -> None:
    """Setup logging configuration"""
    if config_file and os.path.exists(config_file):
        with open(config_file, 'r') as f:
            config = json.load(f)
            logging.config.dictConfig(config)
    else:
        logging.basicConfig(
            level=default_level,
            format='%(asctime)s - %(levelname)s - %(name)s - %(message)s'
        )

def load_config(config_path: Union[str, Path] = "config/config.yaml") -> Dict:
    """Load configuration file (YAML)."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Configuration file not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def get_logger(name: str = "app") -> logging.Logger:
    """Get configured logger instance."""
    logger = logging.getLogger(name)
    if not logger.handlers:
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
    return logger
