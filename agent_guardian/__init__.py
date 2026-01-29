"""
Agent Guardian Detector - Policy Enforcement for CrewAI
"""

__version__ = "0.1.0"

from .enforcer import (
    PolicyEnforcer,
    enable_enforcement,
    disable_enforcement,
    get_active_enforcer,
    get_enforcement_stats
)

from .config import EnforcementConfig, validate_ollama_available

__all__ = [
    # High-level API (recommended)
    'enable_enforcement',
    'disable_enforcement',
    'get_enforcement_stats',
    
    # Advanced API
    'PolicyEnforcer',
    'get_active_enforcer',
    
    # Configuration
    'EnforcementConfig',
    'validate_ollama_available',
    
    # Version
    '__version__'
]