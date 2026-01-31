"""
Policy Discovery Utilities
Auto-discover policy folders and agent role mappings from directory structure
"""

import logging
from pathlib import Path
from typing import Dict, List, Tuple, Set

logger = logging.getLogger(__name__)


class PolicyDiscovery:
    """
    Auto-discovers policy structure from filesystem.
    
    Scans policy directory for:
    1. Policy folders (subdirectories of policy_dir)
    2. Agent roles (subdirectories within policy folders)
    3. Agent role mappings (natural name -> normalized folder name)
    """
    
    def __init__(self, policy_dir: Path):
        """
        Initialize discovery system.
        
        Args:
            policy_dir: Root directory containing policy folders
        """
        self.policy_dir = Path(policy_dir)
        self._cached_folders: List[str] = None
        self._cached_mappings: Dict[str, str] = None
        
    def discover_policy_folders(self) -> List[str]:
        """
        Discover all policy folder names.
        
        Returns:
            List of policy folder names (e.g., ['policies_it_assistant', 'policies_slack'])
        
        Example structure:
            /app/Policies/
            ├── policies_it_assistant/
            ├── policies_slack/
            └── policies_trip_planner/
        """
        if self._cached_folders is not None:
            return self._cached_folders
        
        if not self.policy_dir.exists():
            logger.warning(f"Policy directory not found: {self.policy_dir}")
            self._cached_folders = []
            return []
        
        folders = []
        
        try:
            for item in self.policy_dir.iterdir():
                if item.is_dir() and not item.name.startswith('.'):
                    # Only include directories that look like policy folders
                    # (contain at least one subdirectory - agent role)
                    agent_dirs = [d for d in item.iterdir() if d.is_dir() and not d.name.startswith('.')]
                    if agent_dirs:
                        folders.append(item.name)
                        logger.debug(f"Discovered policy folder: {item.name} (contains {len(agent_dirs)} agent roles)")
            
            self._cached_folders = sorted(folders)
            logger.info(f"Auto-discovered {len(folders)} policy folders: {folders}")
            
        except Exception as e:
            logger.error(f"Error discovering policy folders: {e}")
            self._cached_folders = []
        
        return self._cached_folders
    
    def discover_agent_roles(self) -> Set[str]:
        """
        Discover all unique agent role folder names across all policy folders.
        
        Returns:
            Set of normalized agent role folder names (e.g., {'IT_Support_Diagnostic_Specialist', 'AI_Assistant'})
        """
        if not self.policy_dir.exists():
            return set()
        
        agent_roles = set()
        
        try:
            for policy_folder in self.policy_dir.iterdir():
                if not policy_folder.is_dir() or policy_folder.name.startswith('.'):
                    continue
                
                for agent_dir in policy_folder.iterdir():
                    if agent_dir.is_dir() and not agent_dir.name.startswith('.'):
                        # Check if it contains YAML files (actual policy files)
                        yaml_files = list(agent_dir.glob('*.yaml')) + list(agent_dir.glob('*.yml'))
                        if yaml_files:
                            agent_roles.add(agent_dir.name)
                            logger.debug(f"Discovered agent role: {agent_dir.name} in {policy_folder.name}")
            
            logger.info(f"Auto-discovered {len(agent_roles)} unique agent roles")
            
        except Exception as e:
            logger.error(f"Error discovering agent roles: {e}")
        
        return agent_roles
    
    def generate_role_mappings(self) -> Dict[str, str]:
        """
        Generate agent role mappings from discovered folder structure.
        
        Creates mappings from natural language names to normalized folder names.
        
        Returns:
            Dictionary mapping natural names to folder names
            Example: {
                "Senior Data Analyst": "Senior_Data_Analyst",
                "IT Support Diagnostic Specialist": "IT_Support_Diagnostic_Specialist"
            }
        
        Mapping logic:
            - Replaces underscores with spaces for natural name
            - Handles both directions for flexibility
        """
        if self._cached_mappings is not None:
            return self._cached_mappings
        
        agent_roles = self.discover_agent_roles()
        mappings = {}
        
        for normalized_name in agent_roles:
            # Generate natural language version by replacing underscores with spaces
            natural_name = normalized_name.replace('_', ' ')
            
            # Add bidirectional mapping for flexibility
            mappings[natural_name] = normalized_name
            
            # Also map the normalized name to itself (in case system prompt uses it directly)
            mappings[normalized_name] = normalized_name
            
            logger.debug(f"Generated mapping: '{natural_name}' -> '{normalized_name}'")
        
        self._cached_mappings = mappings
        logger.info(f"Auto-generated {len(agent_roles)} agent role mappings")
        
        return mappings
    
    def discover_all(self) -> Tuple[Dict[str, str], List[str]]:
        """
        Discover both agent role mappings and policy folders.
        
        Returns:
            Tuple of (agent_role_mappings, policy_folders)
        """
        mappings = self.generate_role_mappings()
        folders = self.discover_policy_folders()
        
        return mappings, folders
    
    def clear_cache(self):
        """Clear cached discovery results (useful if filesystem changes)."""
        self._cached_folders = None
        self._cached_mappings = None
        logger.debug("Discovery cache cleared")


def merge_mappings(
    discovered: Dict[str, str],
    custom: Dict[str, str] = None
) -> Dict[str, str]:
    """
    Merge discovered mappings with custom user-provided mappings.
    
    Args:
        discovered: Auto-discovered mappings
        custom: User-provided custom mappings (takes priority)
    
    Returns:
        Merged mappings dictionary
    
    Priority: custom > discovered
    """
    result = discovered.copy()
    
    if custom:
        result.update(custom)
        logger.debug(f"Merged {len(custom)} custom mappings with {len(discovered)} discovered mappings")
    
    return result


def merge_folders(
    discovered: List[str],
    custom: List[str] = None
) -> List[str]:
    """
    Merge discovered policy folders with custom user-provided folders.
    
    Args:
        discovered: Auto-discovered policy folder names
        custom: User-provided custom folder names
    
    Returns:
        Combined list (preserving order, removing duplicates)
    
    Priority: discovered first, then custom additions
    """
    result = discovered.copy()
    
    if custom:
        for folder in custom:
            if folder not in result:
                result.append(folder)
        logger.debug(f"Added {len(custom)} custom folders to {len(discovered)} discovered folders")
    
    return result


# Example usage and testing
if __name__ == "__main__":
    import sys
    
    # Setup logging for testing
    logging.basicConfig(
        level=logging.DEBUG,
        format='[%(levelname)s] [%(name)s] %(message)s'
    )
    
    # Test with example path (modify as needed)
    if len(sys.argv) > 1:
        test_path = Path(sys.argv[1])
    else:
        test_path = Path("./Policies")
    
    print(f"\n{'='*60}")
    print(f"Testing Policy Discovery")
    print(f"{'='*60}\n")
    print(f"Scanning: {test_path.absolute()}\n")
    
    # Initialize discovery
    discovery = PolicyDiscovery(test_path)
    
    # Discover policy folders
    print("Policy Folders:")
    folders = discovery.discover_policy_folders()
    for folder in folders:
        print(f"  - {folder}")
    
    # Discover agent roles
    print("\nAgent Roles:")
    roles = discovery.discover_agent_roles()
    for role in sorted(roles):
        print(f"  - {role}")
    
    # Generate mappings
    print("\nGenerated Mappings:")
    mappings = discovery.generate_role_mappings()
    for natural, normalized in sorted(mappings.items()):
        print(f"  '{natural}' -> '{normalized}'")
    
    # Test merging with custom data
    print("\nTesting Merge with Custom Mappings:")
    custom_mappings = {
        "Custom Agent": "Custom_Agent_Folder",
        "Senior Data Analyst": "Override_Folder"  # Override discovered
    }
    merged = merge_mappings(mappings, custom_mappings)
    print(f"  Merged {len(merged)} total mappings")
    print(f"  Custom override: 'Senior Data Analyst' -> '{merged.get('Senior Data Analyst')}'")
    
    print(f"\n{'='*60}")
    print("Discovery Test Complete!")
    print(f"{'='*60}\n")