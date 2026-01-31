

import logging
import re
import threading
import json
import ast
from datetime import datetime
from typing import Dict, Any, List, Optional
from pathlib import Path

from litellm.integrations.custom_logger import CustomLogger
import yaml

from ..config import EnforcementConfig
from ..storage import SQLiteLogger, PolicyAuditLogger
from ..utils import PathResolver
from ..utils.policy_discovery import PolicyDiscovery, merge_mappings, merge_folders
from .workflow_validator import validate_workflow
from .semantic_validator import get_semantic_validator

logger = logging.getLogger(__name__)

# Logging configuration from environment
import os
LOG_LEVEL = os.environ.get('POLICY_LOG_LEVEL', 'INFO').upper()
ENABLE_DEBUG_PRINTS = LOG_LEVEL == 'DEBUG'
ENABLE_INFO_PRINTS = LOG_LEVEL in ['DEBUG', 'INFO']

def debug_print(message: str, force: bool = False):
    """Print debug messages only if debug logging is enabled."""
    if ENABLE_DEBUG_PRINTS or force:
        print(f"[DEBUG] {message}", flush=True)

def info_print(message: str, force: bool = False):
    """Print info messages only if info logging is enabled."""
    if ENABLE_INFO_PRINTS or force:
        print(f"[INFO] {message}", flush=True)

def warning_print(message: str):
    """Always print warnings."""
    print(f"[WARNING] {message}", flush=True)

def error_print(message: str):
    """Always print errors."""
    print(f"[ERROR] {message}", flush=True)


class SimplePolicyCallback(CustomLogger):
   
    
    def __init__(
        self, 
        config: EnforcementConfig,
        agent_role_mappings: Optional[Dict[str, str]] = None,
        policy_folders: Optional[List[str]] = None,
        enable_auto_discovery: bool = True
    ):
        """
        Initialize policy callback with auto-discovery.
        
        Args:
            config: EnforcementConfig instance
            agent_role_mappings: Optional custom agent role mappings (extends auto-discovered)
            policy_folders: Optional custom policy folder names (extends auto-discovered)
            enable_auto_discovery: Enable automatic discovery (default: True)
        
        Discovery priority:
            1. Auto-discovered mappings/folders (base layer)
            2. User-provided custom mappings/folders (override layer)
        
        Example (zero config):
            callback = SimplePolicyCallback(config)
        
        Example (with custom overrides):
            callback = SimplePolicyCallback(
                config,
                agent_role_mappings={"Special Agent": "Custom_Folder"},
                policy_folders=["legacy_policies"]
            )
        """
        super().__init__()
        
        self.config = config
        
        # ========================================================================
        # AUTO-DISCOVERY SYSTEM
        # ========================================================================
        
        if enable_auto_discovery:
            info_print("🔍 Starting policy structure auto-discovery...", force=True)
            
            try:
                # Initialize discovery system
                discovery = PolicyDiscovery(config.policy_dir)
                
                # Discover structure
                discovered_mappings, discovered_folders = discovery.discover_all()
                
                info_print(f"✅ Auto-discovery complete:", force=True)
                info_print(f"   - Found {len(discovered_folders)} policy folders", force=True)
                info_print(f"   - Generated {len(discovered_mappings)} agent role mappings", force=True)
                
                # Merge with user-provided custom data (custom takes priority)
                self.agent_role_mappings = merge_mappings(discovered_mappings, agent_role_mappings)
                self.policy_folders = merge_folders(discovered_folders, policy_folders)
                
                if agent_role_mappings:
                    info_print(f"   - Added {len(agent_role_mappings)} custom role mappings", force=True)
                if policy_folders:
                    info_print(f"   - Added {len(policy_folders)} custom policy folders", force=True)
                
            except Exception as e:
                error_print(f"Auto-discovery failed: {e}")
                warning_print("Falling back to user-provided mappings only")
                
                # Fallback to user-provided only
                self.agent_role_mappings = agent_role_mappings or {}
                self.policy_folders = policy_folders or []
        else:
            # Auto-discovery disabled - use only user-provided
            info_print("⚠️  Auto-discovery disabled - using only user-provided mappings")
            self.agent_role_mappings = agent_role_mappings or {}
            self.policy_folders = policy_folders or []
        
        # Log final configuration
        info_print(f"📋 Final configuration:", force=True)
        info_print(f"   - {len(self.agent_role_mappings)} agent role mappings", force=True)
        info_print(f"   - {len(self.policy_folders)} policy folders", force=True)
        if ENABLE_DEBUG_PRINTS:
            debug_print(f"   - Policy folders: {self.policy_folders}")
            debug_print(f"   - Agent mappings: {list(self.agent_role_mappings.keys())[:25]}...")
        
        # ========================================================================
        #         
        # Generate unique instance ID
        self.instance_id = f"callback_{id(self)}"
        
        # In-memory session history cache
        self.session_history: Dict[str, List[str]] = {}
        self.session_history_lock = threading.Lock()
        
        # Initialize storage loggers
        self.sqlite_logger = None
        self.audit_logger = None
        
        if config.enable_audit_logging:
            try:
                self.sqlite_logger = SQLiteLogger(config.db_path)
                self.audit_logger = PolicyAuditLogger(config.db_path)
                info_print(f"SQLite logging enabled for {self.instance_id}")
            except Exception as e:
                warning_print(f"SQLite logging disabled due to error: {e}")
                config.enable_audit_logging = False
        
        # Initialize semantic validator
        self.semantic_validator = None
        if config.enable_semantic_validation:
            try:
                self.semantic_validator = get_semantic_validator(
                    ollama_host=config.ollama_host,
                    model=config.semantic_model
                )
                info_print(f"Semantic validation enabled for {self.instance_id}")
            except Exception as e:
                warning_print(f"Semantic validation disabled due to error: {e}")
                config.enable_semantic_validation = False
        
        info_print(f"Policy callback initialized: {self.instance_id}", force=True)

    def _extract_agent_role(self, messages: List[Dict[str, Any]]) -> Optional[str]:
        """
        Extract agent role from system message and normalize using mappings.
        
        """
        full_extracted_role = None
        for message in messages:
            content = message.get('content', '')
            if message.get('role') == 'system' and isinstance(content, str):
                # Pattern 1: "You are a/an [role]"
                match = re.search(r"You are (?:a |an )?(.+?)(?:\.|$|\n)", content, re.IGNORECASE)
                if match:
                    full_extracted_role = match.group(1).strip().split('.')[0].split('\n')[0]
                    break
                
                # Pattern 2: "You're a/an [role]"
                match = re.search(r"You're (?:a |an )?(.+?)(?:\.|$|\n)", content, re.IGNORECASE)
                if match:
                    full_extracted_role = match.group(1).strip().split('.')[0].split('\n')[0]
                    break
        
        if full_extracted_role:
            # Check if there's a mapping for this role
            for role_key, folder_name in self.agent_role_mappings.items():
                if role_key in full_extracted_role:
                    debug_print(f"Mapped role '{full_extracted_role}' -> '{folder_name}'")
                    return folder_name
            
            normalized_role = full_extracted_role.replace(' ', '_')
            debug_print(f"No mapping found for role '{full_extracted_role}', using auto-normalized: '{normalized_role}'")
            return normalized_role
        
        return None

    def _load_specific_policy(self, agent_role_for_folder: str, action_name: str) -> Optional[Dict[str, Any]]:
        """Load the specific policy.yaml for a given agent role and action name."""
        
        print(f"\n{'='*80}")
        print(f"🔍 POLICY SEARCH DEBUG")
        print(f"{'='*80}")
        print(f"Agent Role: {agent_role_for_folder}")
        print(f"Action Name: {action_name}")
        print(f"Policy Base Dir: {self.config.policy_dir}")
        print(f"Policy Folders to Search: {self.policy_folders}")
        print(f"{'='*80}\n")
        
        # Try to find the policy file
        policy_file = PathResolver.find_policy_file(
            policy_dir=self.config.policy_dir,
            agent_role=agent_role_for_folder,
            action_name=action_name,
            policy_folders=self.policy_folders
        )
        
        if policy_file:
            print(f"FOUND POLICY FILE: {policy_file}\n")
            try:
                with open(policy_file, 'r', encoding='utf-8') as f:
                    policy_data = yaml.safe_load(f)
                    print(f"✅ Successfully loaded policy from: {policy_file}\n")
                    return policy_data
            except Exception as e:
                print(f"ERROR loading policy file: {e}\n")
                error_print(f"Failed to load policy from {policy_file}: {e}")
                return None
        else:
            print(f"NO POLICY FILE FOUND")
            print(f"   Searched in:")
            for folder in self.policy_folders:
                search_path = self.config.policy_dir / folder / agent_role_for_folder
                print(f"   - {search_path}")
            print(f"\n")
            return None

    def _extract_function_arguments(self, response_obj: Any, action_name: str = None) -> Dict[str, Any]:
        """Extract function arguments from LLM response with robust parsing."""
        try:
            if not hasattr(response_obj, 'choices') or not response_obj.choices:
                return {}
            
            message = response_obj.choices[0].message
            
            # Priority 1: Modern Tool Calls
            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'arguments'):
                        args_str = tool_call.function.arguments
                        
                        if isinstance(args_str, str):
                            try:
                                return json.loads(args_str, strict=False)
                            except json.JSONDecodeError:
                                try:
                                    return ast.literal_eval(args_str)
                                except:
                                    return {}
                        elif isinstance(args_str, dict):
                            return args_str

            # Priority 2: Legacy/String Content Parsing
            content = getattr(message, 'content', '') or ''
            if content:
                match = re.search(r'##\s*Tool Input:\s*\n\s*"?(\{.+?\})"?', content, re.IGNORECASE | re.DOTALL)
                if not match:
                    match = re.search(r"Action Input:\s*(.+?)(?:\n\n|$)", content, re.IGNORECASE | re.DOTALL)
                
                if match:
                    args_str = match.group(1).strip()
                    
                    # Remove wrapping quotes only
                    if args_str.startswith('"') and args_str.endswith('"'):
                        args_str = args_str[1:-1]
                    
                    # Extract JSON
                    start_idx = args_str.find('{')
                    if start_idx == -1:
                        return {}

                    brace_level = 0
                    end_idx = -1
                    
                    for i, char in enumerate(args_str[start_idx:]):
                        if char == '{':
                            brace_level += 1
                        elif char == '}':
                            brace_level -= 1
                        
                        if brace_level == 0:
                            end_idx = start_idx + i + 1
                            break
                    
                    if end_idx == -1:
                        return {}

                    clean_json_str = args_str[start_idx:end_idx]
                    
                    # Try JSON first, then Python literal eval
                    try:
                        return json.loads(clean_json_str, strict=False)
                    except json.JSONDecodeError:
                        pass

                    try:
                        debug_print(f"JSON parse failed, attempting Python AST eval...")
                        return ast.literal_eval(clean_json_str)
                    except Exception:
                        pass
            
            return {}
        
        except Exception as e:
            warning_print(f"Could not extract function arguments: {e}")
            return {}

    def _extract_action_from_response_obj(self, response_obj: Any) -> tuple[Optional[str], str]:
        """Extract the action name from LiteLLM's response_obj."""
        try:
            if not hasattr(response_obj, 'choices') or not response_obj.choices:
                return None, "no_choices"
            
            message = response_obj.choices[0].message

            if hasattr(message, 'tool_calls') and message.tool_calls:
                for tool_call in message.tool_calls:
                    if hasattr(tool_call, 'function') and hasattr(tool_call.function, 'name'):
                        return tool_call.function.name, "tool_calls"
            
            content = getattr(message, 'content', None)
            if content:
                match = re.search(r"Action:\s*(.+?)(?:\n|$)", content, re.IGNORECASE)
                if match:
                    return match.group(1).strip(), "content_regex"
            
            return None, "no_action_found"
        except Exception as e:
            error_print(f"Exception in _extract_action_from_response_obj: {e}")
            return None, f"exception_{str(e)[:50]}"

    def _check_violations(
        self, 
        session_id: str,
        input_tokens: int,
        output_tokens: int,
        token_limits: Dict[str, int],
        time_limits: Dict[str, str],
        actual_duration_ms: int,
        duration_limits: Dict[str, int],
        function_args: Dict[str, Any],
        pattern_limits: Dict[str, Any],
        agent_role: str,
        action_name: str
    ) -> List[str]:
        """Check for all policy violations and return list of violation messages."""
        violations = []
        failed_structured_fields = set()
        
        # TOKEN LIMIT CHECKS
        if token_limits:
            min_in = token_limits.get('min_input_tokens')
            max_in = token_limits.get('max_input_tokens')
            min_out = token_limits.get('min_output_tokens')
            max_out = token_limits.get('max_output_tokens')
            
            if min_in and input_tokens < min_in:
                violations.append(f"Input tokens ({input_tokens}) below minimum ({min_in})")
            if max_in and input_tokens > max_in:
                violations.append(f"Input tokens ({input_tokens}) exceeds maximum ({max_in})")
            if min_out and output_tokens < min_out:
                violations.append(f"Output tokens ({output_tokens}) below minimum ({min_out})")
            if max_out and output_tokens > max_out:
                violations.append(f"Output tokens ({output_tokens}) exceeds maximum ({max_out})")
        
        # TIME WINDOW CHECKS
        if time_limits:
            current_time_str = datetime.now().strftime('%H:%M:%S')
            min_time_str = time_limits.get('min_hour')
            max_time_str = time_limits.get('max_hour')
            
            if min_time_str and max_time_str:
                try:
                    current_time = datetime.strptime(current_time_str, '%H:%M:%S').time()
                    min_time = datetime.strptime(min_time_str, '%H:%M:%S').time()
                    max_time = datetime.strptime(max_time_str, '%H:%M:%S').time()
                    
                    if not (min_time <= current_time <= max_time):
                        violations.append(
                            f"Execution time ({current_time_str}) is outside allowed window "
                            f"({min_time_str} - {max_time_str})"
                        )
                except Exception as e:
                    debug_print(f"Time check error: {e}")
        
        # DURATION CHECKS
        if duration_limits:
            min_duration = duration_limits.get('min_processing_duration')
            max_duration = duration_limits.get('max_processing_duration')
            
            if min_duration and actual_duration_ms < min_duration:
                violations.append(f"Processing duration ({actual_duration_ms}ms) below minimum ({min_duration}ms)")
            if max_duration and actual_duration_ms > max_duration:
                violations.append(f"Processing duration ({actual_duration_ms}ms) exceeds maximum ({max_duration}ms)")
        
        # FORBIDDEN PATTERN CHECKS
        forbidden_patterns = pattern_limits.get("forbidden_patterns", [])
        if forbidden_patterns:
            for param_name, forbidden_pattern_list in forbidden_patterns.items():
                if param_name in function_args:
                    actual_value = function_args[param_name]
                    if isinstance(actual_value, bool):
                        actual_value = str(actual_value).lower()
                    else:
                        actual_value = str(actual_value)
                    
                    for forbidden_regex in forbidden_pattern_list:
                        try:
                            if re.match(forbidden_regex, actual_value):
                                violations.append(
                                    f"Forbidden pattern violation: '{param_name}' value '{actual_value}' "
                                    f"matches forbidden pattern '{forbidden_regex}'"
                                )
                                failed_structured_fields.add(param_name)
                                break
                        except Exception as e:
                            debug_print(f"Forbidden pattern check error: {e}")
        
        # ALLOWED INPUT PATTERN CHECKS
        input_patterns = pattern_limits.get("input_patterns", [])
        if input_patterns and not any('Forbidden pattern violation' in v for v in violations):
            any_complete_pattern_matched = False
            
            for allowed_pattern_dict in input_patterns:
                this_pattern_matches = True
                
                for param_name, pattern_config in allowed_pattern_dict.items():
                    if param_name not in function_args:
                        this_pattern_matches = False
                        break

                    actual_value = str(function_args[param_name])
                    param_matched = False

                    pattern_list_to_check = pattern_config if isinstance(pattern_config, list) else [str(pattern_config)]

                    for allowed_regex in pattern_list_to_check:
                        try:
                            if re.match(allowed_regex, actual_value):
                                param_matched = True
                                debug_print(f"Pattern matched: {param_name}={actual_value} matches {allowed_regex}")
                                break
                        except Exception as e:
                            debug_print(f"Pattern matching error for {param_name}: {e}")

                    if not param_matched:
                        debug_print(f"Pattern NOT matched: {param_name}={actual_value} against {pattern_list_to_check}")
                        this_pattern_matches = False
                        failed_structured_fields.add(param_name)
                        break
                
                if this_pattern_matches:
                    any_complete_pattern_matched = True
                    debug_print(f"Complete pattern matched!")
                    break
            
            if not any_complete_pattern_matched:
                violations.append(
                    f"Input pattern violation: Function arguments {function_args} "
                    f"did not match any of the allowed patterns."
                )
        
        # SEMANTIC VALIDATION
        if self.config.enable_semantic_validation and self.semantic_validator:
            semantic_checks = pattern_limits.get("semantic_descriptions", {})
            
            if semantic_checks:
                for field_name, semantic_description in semantic_checks.items():
                    if field_name not in function_args:
                        debug_print(f"Skipping semantic validation for {field_name} (field not in function_args)")
                        continue
                    
                    if field_name in failed_structured_fields:
                        debug_print(f"Skipping semantic validation for {field_name} (already failed structured check)")
                        continue
                    
                    field_value = function_args[field_name]
                    debug_print(f"Running semantic validation for field: {field_name}")
                    
                    try:
                        result = self.semantic_validator.validate_field(
                            field_name=field_name,
                            field_value=field_value,
                            semantic_description=semantic_description
                        )
                        
                        if "_semantic_checks" not in function_args:
                            function_args["_semantic_checks"] = {}
                        
                        function_args["_semantic_checks"][field_name] = {
                            "description": semantic_description,
                            "qwen_verdict": "VIOLATION" if result['is_violation'] else "PASS",
                            "qwen_reasoning": result['reasoning'],
                            "duration_ms": result['duration_ms']
                        }
                        
                        if result['is_violation']:
                            violations.append(f"Semantic violation ({field_name}): {result['reasoning']}")
                            warning_print(f"Semantic violation detected for {field_name}")
                        else:
                            info_print(f"Semantic validation passed for {field_name}")
                    except Exception as e:
                        error_print(f"Error during semantic validation: {e}")
                        violations.append(f"Semantic validation system error: {e}")
        
        # WORKFLOW VALIDATION
        if self.config.enable_workflow_validation:
            try:
                execution_history = []
                
                with self.session_history_lock:
                    execution_history = self.session_history.get(session_id, []).copy()
                
                if not execution_history and self.sqlite_logger:
                    debug_print(f"In-memory history for {session_id} is empty, falling back to DB.")
                    execution_history = self.sqlite_logger.get_execution_history(session_id)
                else:
                    debug_print(f"Got execution history from in-memory cache for {session_id}: {execution_history}")
                
                graph_filename = f"{agent_role.replace(' ', '_')}_graph.yaml"
                graph_path = None

                for policy_folder in self.policy_folders:
                    test_path = self.config.policy_dir / policy_folder / agent_role / graph_filename
                    if test_path.exists():
                        graph_path = test_path
                        debug_print(f"Found workflow graph for {agent_role} at: {graph_path}")
                        break

                if graph_path:
                    debug_print(f"Checking workflow for {agent_role} - History: {execution_history}, Next: {action_name}")
                    
                    is_valid, error_msg = validate_workflow(
                        agent_role=agent_role,
                        action_to_execute=action_name,
                        execution_history=execution_history,
                        graph_yaml_path=graph_path
                    )
                    
                    if not is_valid:
                        violations.append(f"Workflow sequence violation: {error_msg}")
                        warning_print(f"Workflow Graph FAILED for {agent_role} -> {action_name}")
                    else:
                        info_print(f"Workflow Graph PASSED for {agent_role} -> {action_name}")
                else:
                    info_print(f"No control flow graph found for {agent_role}, skipping workflow validation")
                    
            except Exception as e:
                error_print(f"Error during workflow validation: {e}")
                violations.append(f"Workflow validation system error: {e}")
                
        return violations

    def _extract_token_limits(self, policy_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract token limits from policy data."""
        token_limits = {}
        try:
            numeric_limits = policy_data.get('input_restrictions', {}).get('numeric_limits', {})
            
            if 'input_tokens' in numeric_limits:
                input_tokens = numeric_limits['input_tokens']
                token_limits['min_input_tokens'] = input_tokens.get('min_input_tokens')
                token_limits['max_input_tokens'] = input_tokens.get('max_input_tokens')
                
            if 'output_tokens' in numeric_limits:
                output_tokens = numeric_limits['output_tokens']
                token_limits['max_output_tokens'] = output_tokens.get('max_output_tokens')
                
        except Exception as e:
            debug_print(f"Could not extract token limits from policy: {e}")
        
        return token_limits
    
    def _extract_time_limits(self, policy_data: Dict[str, Any]) -> Dict[str, str]:
        """Extract time restrictions from policy data."""
        time_limits = {}
        try:
            numeric_limits = policy_data.get('input_restrictions', {}).get('numeric_limits', {})
            
            if 'time' in numeric_limits:
                time_restrictions = numeric_limits['time']
                time_limits['min_hour'] = time_restrictions.get('min_hour')
                time_limits['max_hour'] = time_restrictions.get('max_hour')
                
        except Exception as e:
            debug_print(f"Could not extract time limits from policy: {e}")
        
        return time_limits
    
    def _extract_duration_limits(self, policy_data: Dict[str, Any]) -> Dict[str, int]:
        """Extract duration restrictions from policy data."""
        duration_limits = {}
        try:
            numeric_limits = policy_data.get('input_restrictions', {}).get('numeric_limits', {})
            
            if 'duration' in numeric_limits:
                duration_restrictions = numeric_limits['duration']
                duration_limits['min_processing_duration'] = duration_restrictions.get('min_processing_duration')
                duration_limits['max_processing_duration'] = duration_restrictions.get('max_processing_duration')
                
        except Exception as e:
            debug_print(f"Could not extract duration limits from policy: {e}")
        
        return duration_limits

    def _extract_pattern_limits(self, policy_data: Dict[str, Any]) -> Dict[str, Any]:
        """Extract pattern restrictions from policy data."""
        pattern_limits = {"input_patterns": [], "forbidden_patterns": [], "semantic_descriptions": {}}
        
        try:
            input_restrictions = policy_data.get('input_restrictions', {})
            categorical_values = input_restrictions.get('categorical_values', [])
            if isinstance(categorical_values, dict): 
                categorical_values = [categorical_values]
            
            for item in categorical_values:
                if isinstance(item, dict):
                    for key, value in item.items():
                        if "input_pattern" in key:
                            pattern_limits["input_patterns"].append(value)
                            
            forbidden_patterns_list = input_restrictions.get('forbidden_patterns', [])
            if isinstance(forbidden_patterns_list, dict): 
                forbidden_patterns_list = [forbidden_patterns_list]
                
            for pattern_item in forbidden_patterns_list:
                if isinstance(pattern_item, dict) and 'pattern' in pattern_item:
                    pattern_limits["forbidden_patterns"].append(pattern_item['pattern'])
            
            semantic_descriptions = input_restrictions.get('semantic_descriptions', {})
            if isinstance(semantic_descriptions, dict):
                pattern_limits["semantic_descriptions"] = semantic_descriptions
                debug_print(f"Extracted semantic descriptions: {list(semantic_descriptions.keys())}")
                            
        except Exception as e:
            debug_print(f"Could not extract pattern limits from policy: {e}")
        
        return pattern_limits

    def _count_tokens(self, messages: List[Dict[str, Any]]) -> int:
        """Simple token counting - approximate by counting words * 1.3."""
        total_text = ""
        for message in messages:
            content = message.get('content', '')
            if isinstance(content, str):
                total_text += content + " "
        
        word_count = len(total_text.split())
        estimated_tokens = int(word_count * 1.3)
        return estimated_tokens
    
    def _count_output_tokens(self, response_obj: Any) -> int:
        """Count tokens in the LLM response."""
        try:
            if not hasattr(response_obj, 'choices') or not response_obj.choices:
                return 0
            
            message = response_obj.choices[0].message
            content = getattr(message, 'content', '') or ''
            
            word_count = len(content.split())
            estimated_tokens = int(word_count * 1.3)
            return estimated_tokens
            
        except Exception as e:
            debug_print(f"Could not count output tokens: {e}")
            return 0

    def _generate_session_id(self, agent_role: str) -> str:
        """Generate persistent session ID for entire crew execution."""
        if not hasattr(self, '_persistent_session_id'):
            start_time = datetime.now().strftime('%Y%m%d_%H%M%S')
            self._persistent_session_id = f"session_{start_time}_{self.instance_id}"
        
        return self._persistent_session_id

    def log_success_event(self, kwargs: Dict[str, Any], response_obj: Any, start_time: float, end_time: float):
        """Log successful requests with policy enforcement and audit."""
        if not ENABLE_INFO_PRINTS:
            return
            
        info_print(f"Processing sync success event for {self.instance_id}")
        
        messages = kwargs.get('messages', [])
        agent_role = self._extract_agent_role(messages)
        action_name, extraction_method = self._extract_action_from_response_obj(response_obj)
        
        session_id = self._generate_session_id(agent_role)
        
        if not agent_role:
            debug_print(f"Could not extract agent_role, skipping policy check")
            return

        if not action_name:
            debug_print(f"No tool found for {agent_role}, skipping policy check (thought step)")
            return
        
        debug_print(f"Policy check for action: {action_name}")
        
        policy_data = self._load_specific_policy(agent_role, action_name)
        
        if policy_data:
            try:
                # Calculate duration
                if isinstance(start_time, (int, float)) and isinstance(end_time, (int, float)):
                    duration_seconds = end_time - start_time
                else:
                    try:
                        duration_seconds = float(end_time) - float(start_time)
                    except (TypeError, ValueError):
                        duration_seconds = (end_time - start_time).total_seconds()
                
                duration_ms = int(duration_seconds * 1000)
                
                input_tokens = self._count_tokens(messages)
                output_tokens = self._count_output_tokens(response_obj)
                function_args = self._extract_function_arguments(response_obj, action_name)
                
                token_limits = self._extract_token_limits(policy_data)
                time_limits = self._extract_time_limits(policy_data)
                duration_limits = self._extract_duration_limits(policy_data)
                pattern_limits = self._extract_pattern_limits(policy_data)
                
                violations = self._check_violations(
                    session_id, input_tokens, output_tokens, token_limits, 
                    time_limits, duration_ms, duration_limits,
                    function_args, pattern_limits,
                    agent_role, action_name
                )
                
                # Prepare audit data
                violation_text = " | ".join(violations) if violations else ""
                
                audit_data = {
                    'timestamp': datetime.now().isoformat()[:19],
                    'agent_role': agent_role,
                    'action_name': action_name,
                    'input_tokens': input_tokens,
                    'output_tokens': output_tokens,
                    'processing_duration_ms': duration_ms,
                    'execution_time': datetime.now().strftime('%H:%M:%S'),
                    'function_args': function_args,
                    'policy_passed': len(violations) == 0,
                    'violation_reasons': violation_text,
                    'session_id': session_id,
                    'callback_instance_id': self.instance_id
                }
                
                # Log to audit database
                if self.audit_logger:
                    self.audit_logger.log_policy_audit(audit_data)
                
                if violations:
                    print(f"\n!! POLICY VIOLATIONS DETECTED!!:", flush=True)
                    for violation in violations:
                        print(f" {violation}", flush=True)
                else:
                    info_print("NO VIOLATIONS DETECTED", force=True)
                
            except Exception as e:
                error_print(f"Error in violation checking: {e}")
        
        # Log to SQLite
        if self.sqlite_logger and policy_data:
            self.sqlite_logger.log_activity(session_id, agent_role, action_name, function_args, [])
            
            # Append to in-memory history cache
            with self.session_history_lock:
                if session_id not in self.session_history:
                    self.session_history[session_id] = []
                self.session_history[session_id].append(action_name)
                debug_print(f"Appended to in-memory history for {session_id}: {action_name}")

    def get_stats(self) -> Dict[str, Any]:
        """
        Get statistics about policy enforcement.
        
        🔍 NEW: Includes auto-discovery information.
        """
        stats = {
            "callback_instance_id": self.instance_id,
            "logging_enabled": self.config.enable_audit_logging,
            "semantic_validation_enabled": self.config.enable_semantic_validation,
            "workflow_validation_enabled": self.config.enable_workflow_validation,
            "auto_discovery_info": {
                "agent_role_mappings_count": len(self.agent_role_mappings),
                "policy_folders_count": len(self.policy_folders),
                "discovered_policy_folders": self.policy_folders
            }
        }
        
        if self.sqlite_logger:
            try:
                sqlite_stats = self.sqlite_logger.get_activity_stats()
                stats["sqlite_stats"] = sqlite_stats
            except Exception as e:
                stats["sqlite_error"] = str(e)
        
        return stats

    def reset_session_id(self):
        """Clear the persistent session ID from the callback instance."""
        if hasattr(self, '_persistent_session_id'):
            try:
                with self.session_history_lock:
                    if self._persistent_session_id in self.session_history:
                        del self.session_history[self._persistent_session_id]
                        debug_print(f"Cleared in-memory history for old session {self._persistent_session_id}")
                
                del self._persistent_session_id
                info_print(f"Callback {self.instance_id}: Persistent session ID has been reset.")
            except Exception as e:
                error_print(f"Callback {self.instance_id}: Error resetting session ID: {e}")
        else:
            info_print(f"Callback {self.instance_id}: No session ID to reset. Ready for new crew.")
            
            with self.session_history_lock:
                self.session_history = {}
                debug_print("Cleared all in-memory history as a fallback.")

   # FIXED VERSION - Find and replace the clear_execution_history method in your policy_callback.py

    def clear_execution_history(self):
        """
        Clear execution history for the current session without resetting session ID.
        
        CRITICAL FIX: Now clears BOTH in-memory cache AND database history to prevent
        cross-contamination between batch processing runs.
        """
        if hasattr(self, '_persistent_session_id'):
            session_id = self._persistent_session_id
            
            # Clear in-memory cache
            with self.session_history_lock:
                if session_id in self.session_history:
                    action_count = len(self.session_history[session_id])
                    self.session_history[session_id] = []
                    info_print(f"Cleared {action_count} actions from in-memory history (session: {session_id})")
                else:
                    self.session_history[session_id] = []
                    info_print(f"Initialized empty in-memory history (session: {session_id})")
            
            # CRITICAL: Also clear database history to prevent cross-contamination in batch mode
            # Without this, workflow validation reads old history from DB when in-memory is empty
            if self.sqlite_logger:
                try:
                    import sqlite3
                    with sqlite3.connect(str(self.sqlite_logger.db_path), timeout=30.0) as conn:
                        cursor = conn.execute(
                            "DELETE FROM agent_activities WHERE session_id = ?",
                            (session_id,)
                        )
                        rows_deleted = cursor.rowcount
                        conn.commit()
                        info_print(f"Cleared {rows_deleted} database records for session: {session_id}")
                except Exception as e:
                    error_print(f"Failed to clear database history: {e}")
        else:
            with self.session_history_lock:
                self.session_history = {}
            info_print("No active session - cleared all execution history")