"""
Path Resolution Utilities
Handles Docker vs local environment path differences
WINDOWS COMPATIBLE VERSION
"""

import os
from pathlib import Path
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class PathResolver:
    """Resolves paths for both Docker and local environments (Windows-compatible)."""
    
    @staticmethod
    def is_docker_environment() -> bool:
        """
        Check if running inside Docker container.
        Works on Windows Docker Desktop (using WSL2 or Hyper-V).
        """
        return (
            os.path.exists('/.dockerenv') or  # Linux containers
            os.path.exists('C:\\.dockerenv') or  # Windows containers (rare)
            os.getenv('DOCKER_CONTAINER') == 'true'  # Explicit env var
        )
    
    @staticmethod
    def resolve_policy_dir(user_path: Optional[str] = None) -> Path:
        r"""
        Resolve the policy directory path.
        
        Args:
            user_path: User-provided path (overrides auto-detection)
                      Can be: "C:\\policies", r"C:\policies", or "./policies"
        
        Returns:
            Resolved Path object (handles Windows paths automatically)
        
        Examples:
            Windows absolute path:
                path = PathResolver.resolve_policy_dir(r"C:\policies")
            
            Relative path (works on Windows and Linux):
                path = PathResolver.resolve_policy_dir("./policies")
            
            Auto-detect:
                path = PathResolver.resolve_policy_dir()
        """
        if user_path:
            path = Path(user_path).resolve()
            logger.debug(f"Using user-provided policy path: {path}")
            return path
        
        # Auto-detect based on environment
        if PathResolver.is_docker_environment():
            path = Path('/app/policies')
            logger.debug(f"Docker environment: Using {path}")
        else:
            # Windows/Local: Use current working directory
            path = Path.cwd() / 'policies'
            logger.debug(f"Local environment: Using {path}")
        
        return path
    
    @staticmethod
    def resolve_db_path(user_path: Optional[str] = None) -> Path:
        r"""
        Resolve the database file path.
        
        Args:
            user_path: User-provided path (overrides auto-detection)
        
        Returns:
            Resolved Path object
        
        Examples:
            Windows absolute path:
                db = PathResolver.resolve_db_path(r"C:\data\enforcement.db")
            
            Relative path:
                db = PathResolver.resolve_db_path("./enforcement.db")
        """
        if user_path:
            path = Path(user_path).resolve()
            logger.debug(f"Using user-provided DB path: {path}")
            return path
        
        # Auto-detect based on environment
        if PathResolver.is_docker_environment():
            path = Path('/app/data/enforcement.db')
            logger.debug(f"Docker environment: Using {path}")
        else:
            # Windows/Local: Use current working directory
            path = Path.cwd() / 'enforcement.db'
            logger.debug(f"Local environment: Using {path}")
        
        return path
    
    @staticmethod
    def find_policy_file(
        policy_dir: Path,
        agent_role: str,
        action_name: str,
        policy_folders: list
    ) -> Optional[Path]:
        """
        Find a policy file across multiple policy folder structures.
        """
        # Sanitize action name (remove characters that are invalid in Windows filenames)
        sanitized_action = action_name.replace("'", '_').replace(' ', '_')
        sanitized_action = sanitized_action.replace(':', '_').replace('*', '_')
        sanitized_action = sanitized_action.replace('?', '_').replace('"', '_')
        sanitized_action = sanitized_action.replace('<', '_').replace('>', '_')
        sanitized_action = sanitized_action.replace('|', '_')
        
        print(f"   📂 Sanitized action name: '{action_name}' -> '{sanitized_action}'")
        
        for folder in policy_folders:
            base_path = policy_dir / folder / agent_role
            
            print(f"   📂 Checking: {base_path}")
            
            if not base_path.is_dir():
                print(f"      ❌ Directory does not exist")
                continue
            
            print(f"      ✓ Directory exists")
            
            # Try flat structure first
            flat_file = base_path / f"{sanitized_action}.yaml"
            print(f"      📄 Looking for flat file: {flat_file.name}")
            
            if flat_file.is_file():
                print(f"      ✅ FOUND (flat structure): {flat_file}")
                logger.debug(f"Found policy (flat): {flat_file}")
                return flat_file
            else:
                print(f"      ❌ Not found as flat file")
            
            # Try nested structure
            nested_folder = base_path / action_name
            print(f"      📁 Looking for nested folder: {nested_folder.name}")
            
            if nested_folder.is_dir():
                print(f"      ✓ Nested folder exists")
                yaml_files = list(nested_folder.glob('*.yaml'))
                print(f"      📄 YAML files in folder: {[f.name for f in yaml_files]}")
                
                if yaml_files:
                    print(f"      ✅ FOUND (nested structure): {yaml_files[0]}")
                    logger.debug(f"Found policy (nested): {yaml_files[0]}")
                    return yaml_files[0]
            else:
                print(f"      ❌ Nested folder does not exist")
        
        print(f"   ❌ Policy file not found in any location")
        return None