"""
Configuration utilities for the Project Proposal Summarizer
"""

import json
import yaml
import os


def load_config(config_type="json"):
    """
    Load configuration from file

    Args:
        config_type (str): Type of config file to load ("json" or "yaml")

    Returns:
        dict: Configuration settings
    """
    if config_type.lower() == "json":
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "config.json")
        with open(config_path, 'r') as f:
            return json.load(f)
    elif config_type.lower() == "yaml":
        config_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "settings.yaml")
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    else:
        raise ValueError(f"Unsupported config type: {config_type}")


def get_agent_config(agent_name):
    """
    Get configuration for a specific agent

    Args:
        agent_name (str): Name of the agent

    Returns:
        dict: Agent configuration
    """
    config = load_config()
    return config.get("agents", {}).get(agent_name, {})


def get_logging_config():
    """
    Get logging configuration

    Returns:
        dict: Logging configuration
    """
    config = load_config()
    return config.get("logging", {})


def get_model_config(model_name=None):
    """
    Get model configuration

    Args:
        model_name (str, optional): Name of the model

    Returns:
        dict: Model configuration
    """
    config = load_config()
    models = config.get("model", {})

    if model_name and model_name in models:
        return models[model_name]

    return models