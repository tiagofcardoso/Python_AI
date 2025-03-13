"""
System information utilities for the Project Proposal Summarizer
"""

import os
import platform
import sys
import socket
import uuid
import psutil


def get_system_info():
    """
    Get detailed system information

    Returns:
        dict: System information
    """
    system_info = {
        'hostname': socket.gethostname(),
        'platform': platform.platform(),
        'python_version': sys.version,
        'processor': platform.processor(),
        'architecture': platform.architecture(),
        'machine': platform.machine(),
        'system': platform.system(),
        'release': platform.release(),
        'version': platform.version()
    }

    # Add CPU information if psutil is available
    try:
        system_info['cpu_count_physical'] = psutil.cpu_count(logical=False)
        system_info['cpu_count_logical'] = psutil.cpu_count(logical=True)
        system_info['cpu_freq'] = psutil.cpu_freq()._asdict() if psutil.cpu_freq() else None
        system_info['memory_total'] = psutil.virtual_memory().total
        system_info['memory_available'] = psutil.virtual_memory().available
    except (ImportError, AttributeError):
        system_info['cpu_count'] = os.cpu_count() if hasattr(os, 'cpu_count') else None

    # Add unique identifiers
    system_info['process_id'] = str(uuid.uuid4())
    system_info['timestamp'] = __import__('datetime').datetime.now().isoformat()

    return system_info


def print_system_info():
    """
    Print formatted system information to console
    """
    info = get_system_info()

    print("\n===== System Information =====")
    for key, value in info.items():
        print(f"{key.replace('_', ' ').title()}: {value}")
    print("=============================\n")