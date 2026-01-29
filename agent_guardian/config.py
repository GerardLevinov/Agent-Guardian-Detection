"""
Configuration Management for Agent Guardian Detector
Handles both programmatic and environment-based configuration
WINDOWS COMPATIBLE VERSION
"""

import os
from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import logging

logger = logging.getLogger(__name__)


@dataclass
class EnforcementConfig:
    """
    Configuration for policy enforcement.
    
    Attributes:
        policy_dir: Path to policies folder (contains agent-specific subfolders)
        db_path: Path to SQLite database for audit logging
        semantic_model: Ollama model for semantic validation
        ollama_host: Ollama server URL (auto-detected if None)
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR)
        enable_workflow_validation: Enable control flow graph validation
        enable_semantic_validation: Enable LLM-based semantic validation
        enable_audit_logging: Enable detailed audit trail in SQLite
    """
    
    policy_dir: Path
    db_path: Path
    semantic_model: str = "qwen2.5:7b-instruct"
    ollama_host: Optional[str] = None
    log_level: str = "INFO"
    enable_workflow_validation: bool = True
    enable_semantic_validation: bool = True
    enable_audit_logging: bool = True
    
    def __post_init__(self):
        """Validate and normalize configuration after initialization."""
        
        # Convert strings to Path objects (Path handles Windows paths automatically)
        self.policy_dir = Path(self.policy_dir).resolve()
        self.db_path = Path(self.db_path).resolve()
        
        # Auto-detect Ollama host if not provided
        if self.ollama_host is None:
            self.ollama_host = self._detect_ollama_host()
        
        # Validate policy directory exists
        if not self.policy_dir.exists():
            raise FileNotFoundError(
                f"Policy directory not found: {self.policy_dir}\n"
                f"Please create it or provide a valid path.\n"
                f"Example: policy_dir=r'C:\\policies' or policy_dir='./policies'"
            )
        
        # Ensure database directory exists (works on Windows)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Set up logging
        self._configure_logging()
        
        logger.info(f"Agent Guardian configured:")
        logger.info(f"  Policy Dir: {self.policy_dir}")
        logger.info(f"  Database: {self.db_path}")
        logger.info(f"  Ollama: {self.ollama_host}")
    
    def _detect_ollama_host(self) -> str:
        """Auto-detect Ollama host based on environment."""
        
        # Check if running in Docker (Windows Docker uses WSL)
        in_docker = (
            os.path.exists('/.dockerenv') or  # Linux containers
            os.path.exists('C:\\.dockerenv') or  # Windows containers (rare)
            os.getenv('DOCKER_CONTAINER') == 'true'  # Env var detection
        )
        
        if in_docker:
            # In Docker, try to connect to host
            host = 'http://host.docker.internal:11434'
            logger.info(f"Docker environment detected, using Ollama host: {host}")
        else:
            # On Windows local machine
            host = 'http://localhost:11434'
            logger.info(f"Windows environment detected, using Ollama host: {host}")
        
        return host
    
    def _configure_logging(self):
        """Configure logging based on log_level."""
        numeric_level = getattr(logging, self.log_level.upper(), logging.INFO)
        logging.basicConfig(
            level=numeric_level,
            format='[%(levelname)s] [%(name)s] %(message)s'
        )
    
    @classmethod
    def from_env(cls, **overrides):
        r"""
        Create configuration from environment variables.
        
        Environment variables:
            POLICY_DIR: Path to policies folder
            ENFORCEMENT_DB: Path to SQLite database
            SEMANTIC_MODEL: Ollama model name
            OLLAMA_HOST: Ollama server URL
            LOG_LEVEL: Logging level
        
        Args:
            **overrides: Override specific values programmatically
        
        Returns:
            EnforcementConfig instance
        
        Example (Windows):
            Set env var in PowerShell:
                $env:POLICY_DIR = "C:\\policies"
            
            Or in Python:
                config = EnforcementConfig.from_env(
                    policy_dir=r"C:\policies"
                )
        """
        
        # Detect if in Docker for smart defaults
        in_docker = (
            os.path.exists('/.dockerenv') or
            os.getenv('DOCKER_CONTAINER') == 'true'
        )
        
        # Windows-friendly defaults
        if in_docker:
            default_policy_dir = '/app/policies'
            default_db_path = '/app/data/enforcement.db'
        else:
            # Use current working directory on Windows
            default_policy_dir = str(Path.cwd() / 'policies')
            default_db_path = str(Path.cwd() / 'enforcement.db')
        
        config = {
            'policy_dir': os.getenv('POLICY_DIR', default_policy_dir),
            'db_path': os.getenv('ENFORCEMENT_DB', default_db_path),
            'semantic_model': os.getenv('SEMANTIC_MODEL', 'qwen2.5:7b-instruct'),
            'ollama_host': os.getenv('OLLAMA_HOST'),  # None = auto-detect
            'log_level': os.getenv('LOG_LEVEL', 'INFO'),
        }
        
        # Apply overrides
        config.update(overrides)
        
        return cls(**config)


def validate_ollama_available(config: EnforcementConfig) -> None:
    """
    Validate that Ollama is available at the configured host.
    Fails fast with clear error message if not available.
    
    Args:
        config: EnforcementConfig instance
    
    Raises:
        ConnectionError: If Ollama is not reachable
    
    Note (Windows):
        Make sure Ollama Desktop app is running or you've started 'ollama serve'
    """
    import requests
    
    try:
        response = requests.get(f"{config.ollama_host}/api/tags", timeout=5)
        if response.status_code == 200:
            logger.info(f"✓ Ollama is available at {config.ollama_host}")
            
            # Check if the required model is available
            models = response.json().get('models', [])
            model_names = [m['name'] for m in models]
            
            if config.semantic_model not in model_names:
                logger.warning(
                    f"⚠️  Model '{config.semantic_model}' not found in Ollama.\n"
                    f"Available models: {', '.join(model_names)}\n"
                    f"Run in PowerShell: ollama pull {config.semantic_model}"
                )
        else:
            raise ConnectionError(f"Ollama returned status {response.status_code}")
            
    except requests.exceptions.RequestException as e:
        raise ConnectionError(
            f"❌ Cannot connect to Ollama at {config.ollama_host}\n"
            f"Error: {e}\n\n"
            f"Please ensure Ollama is running:\n"
            f"  - Windows: Start Ollama Desktop app or run 'ollama serve' in PowerShell\n"
            f"  - Download: https://ollama.com/download/windows\n"
            f"  - Docker: Ensure ollama service is in docker-compose.yml\n"
        )