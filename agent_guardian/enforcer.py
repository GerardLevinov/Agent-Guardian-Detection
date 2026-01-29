"""
High-Level API for Agent Guardian Detector
The "easy button" for policy enforcement
"""

import logging
from typing import Optional, Dict, List
from pathlib import Path

import litellm

from .config import EnforcementConfig, validate_ollama_available
from .core.policy_callback import SimplePolicyCallback

logger = logging.getLogger(__name__)

# Global state
_active_enforcer: Optional['PolicyEnforcer'] = None
_callback_instance: Optional[SimplePolicyCallback] = None


class PolicyEnforcer:
    """
    High-level policy enforcer that manages the entire enforcement lifecycle.
    
    Usage:
        # Simple setup
        enforcer = PolicyEnforcer(
            policy_dir="./policies",
            db_path="./enforcement.db"
        )
        
        # Your CrewAI code runs normally
        crew = Crew(agents=[...], tasks=[...])
        result = crew.kickoff()
        
        # Check stats
        stats = enforcer.get_stats()
    """
    
    def __init__(
        self,
        policy_dir: Optional[str] = None,
        db_path: Optional[str] = None,
        semantic_model: str = "qwen2.5:7b-instruct",
        ollama_host: Optional[str] = None,
        log_level: str = "INFO",
        enable_workflow_validation: bool = True,
        enable_semantic_validation: bool = True,
        enable_audit_logging: bool = True,
        auto_register: bool = True,
        agent_role_mappings: Optional[Dict[str, str]] = None,
        policy_folders: Optional[List[str]] = None,
        validate_ollama: bool = True
    ):
        """
        Initialize policy enforcer.
        
        Args:
            policy_dir: Path to policies folder (auto-detected if None)
            db_path: Path to SQLite database (auto-detected if None)
            semantic_model: Ollama model for semantic validation
            ollama_host: Ollama server URL (auto-detected if None)
            log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
            enable_workflow_validation: Enable control flow graph validation
            enable_semantic_validation: Enable LLM-based semantic validation
            enable_audit_logging: Enable detailed audit trail in SQLite
            auto_register: Automatically register with LiteLLM callbacks
            agent_role_mappings: Custom agent role mappings (extends defaults)
            policy_folders: Custom policy folder names (extends defaults)
            validate_ollama: Check Ollama availability on startup
        
        Example:
            # Minimal - auto-detects everything
            enforcer = PolicyEnforcer()
            
            # Custom paths
            enforcer = PolicyEnforcer(
                policy_dir=r"C:\\my_policies",
                db_path="./my_enforcement.db"
            )
            
            # Disable semantic validation (no Ollama required)
            enforcer = PolicyEnforcer(
                enable_semantic_validation=False
            )
        """
        
        # Create configuration
        if policy_dir or db_path:
            # User provided paths
            config_kwargs = {}
            if policy_dir:
                config_kwargs['policy_dir'] = policy_dir
            if db_path:
                config_kwargs['db_path'] = db_path
            
            self.config = EnforcementConfig.from_env(**config_kwargs)
        else:
            # Auto-detect everything
            self.config = EnforcementConfig.from_env()
        
        # Apply additional settings
        self.config.semantic_model = semantic_model
        if ollama_host:
            self.config.ollama_host = ollama_host
        self.config.log_level = log_level
        self.config.enable_workflow_validation = enable_workflow_validation
        self.config.enable_semantic_validation = enable_semantic_validation
        self.config.enable_audit_logging = enable_audit_logging
        
        # Validate Ollama if semantic validation is enabled
        if enable_semantic_validation and validate_ollama:
            try:
                validate_ollama_available(self.config)
            except ConnectionError as e:
                logger.error(str(e))
                logger.warning("Disabling semantic validation due to Ollama unavailability")
                self.config.enable_semantic_validation = False
        
        # Create policy callback
        self.callback = SimplePolicyCallback(
            config=self.config,
            agent_role_mappings=agent_role_mappings,
            policy_folders=policy_folders
        )
        
        # Auto-register with LiteLLM if requested
        if auto_register:
            self.register()
        
        logger.info("PolicyEnforcer initialized successfully")
    
    def register(self):
        """Register the callback with LiteLLM."""
        global _callback_instance
        
        # Clear existing callbacks of our type
        litellm.success_callback = [
            cb for cb in litellm.success_callback 
            if not isinstance(cb, SimplePolicyCallback)
        ]
        litellm.failure_callback = [
            cb for cb in litellm.failure_callback 
            if not isinstance(cb, SimplePolicyCallback)
        ]
        if hasattr(litellm, 'input_callback'):
            litellm.input_callback = [
                cb for cb in litellm.input_callback 
                if not isinstance(cb, SimplePolicyCallback)
            ]
        
        # Register our callback
        litellm.success_callback.append(self.callback)
        litellm.failure_callback.append(self.callback)
        litellm.callbacks = [self.callback]
        
        if hasattr(litellm, 'input_callback'):
            litellm.input_callback.append(self.callback)
        
        _callback_instance = self.callback
        
        logger.info(f"Policy callback registered: {self.callback.instance_id}")
    
    def unregister(self):
        """Unregister the callback from LiteLLM."""
        litellm.success_callback = [
            cb for cb in litellm.success_callback 
            if cb != self.callback
        ]
        litellm.failure_callback = [
            cb for cb in litellm.failure_callback 
            if cb != self.callback
        ]
        litellm.callbacks = [
            cb for cb in litellm.callbacks 
            if cb != self.callback
        ]
        if hasattr(litellm, 'input_callback'):
            litellm.input_callback = [
                cb for cb in litellm.input_callback 
                if cb != self.callback
            ]
        
        logger.info("Policy callback unregistered")
    
    def get_stats(self) -> Dict:
        """Get enforcement statistics."""
        return self.callback.get_stats()
    
    def reset_session(self):
        """Reset session ID for new crew execution."""
        self.callback.reset_session_id()
    
    def clear_history(self):
        """Clear execution history without resetting session."""
        self.callback.clear_execution_history()


def enable_enforcement(
    policy_dir: Optional[str] = None,
    db_path: Optional[str] = None,
    **kwargs
) -> PolicyEnforcer:
    """
    One-line function to enable policy enforcement.
    
    This is the simplest way to add enforcement to your CrewAI app.
    
    Args:
        policy_dir: Path to policies folder (auto-detected if None)
        db_path: Path to database file (auto-detected if None)
        **kwargs: Additional arguments passed to PolicyEnforcer
    
    Returns:
        PolicyEnforcer instance
    
    Example:
        # Simplest usage - auto-detects everything
        from agent_guardian import enable_enforcement
        
        enable_enforcement()
        
        # Your normal CrewAI code
        crew = Crew(agents=[...], tasks=[...])
        result = crew.kickoff()
    
    Example with custom paths:
        enable_enforcement(
            policy_dir="./my_policies",
            db_path="./my_db.sqlite"
        )
    """
    global _active_enforcer
    
    enforcer = PolicyEnforcer(
        policy_dir=policy_dir,
        db_path=db_path,
        **kwargs
    )
    
    _active_enforcer = enforcer
    return enforcer


def disable_enforcement():
    """Disable policy enforcement (unregister callbacks)."""
    global _active_enforcer
    
    if _active_enforcer:
        _active_enforcer.unregister()
        _active_enforcer = None
        logger.info("Policy enforcement disabled")


def get_active_enforcer() -> Optional[PolicyEnforcer]:
    """Get the currently active enforcer instance."""
    return _active_enforcer


def get_enforcement_stats() -> Dict:
    """
    Get statistics from the active enforcer.
    
    Returns:
        Dictionary with enforcement statistics
    
    Raises:
        RuntimeError: If no enforcer is active
    """
    if not _active_enforcer:
        raise RuntimeError(
            "No active enforcer. Call enable_enforcement() first."
        )
    
    return _active_enforcer.get_stats()