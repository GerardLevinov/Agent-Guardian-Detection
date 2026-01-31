"""
Workflow Validation Module for Control Flow Graph Checks
=========================================================
VERSION 2.2: Supports BOTH old and new graph formats

Old format: required_leading_contexts as list of lists
New format: required_leading_contexts with 'context', 'occurrence_count', 'percentage'

The enforcer only needs the pattern - occurrence_count and percentage are ignored.

MOVED TO CORE MODULE - Windows Compatible
"""

from typing import List, Dict, Any, Optional, Tuple
import yaml
from pathlib import Path


def normalize_pattern(pattern: Any) -> List[Dict[str, Any]]:
    """
    Normalize a pattern to standard format.
    Handles both old format (list of dicts) and new format (dict with 'context' key).
    
    Args:
        pattern: Either a list of action dicts or a dict with 'context' key
        
    Returns:
        List of action dicts in standard format: [{'action_name': ..., 'repeat': ...}, ...]
    
    Examples:
        Old format input: [{'action_name': 'X', 'repeat': False}, ...]
        Returns: [{'action_name': 'X', 'repeat': False}, ...]
        
        New format input: {'context': [...], 'occurrence_count': 5, 'percentage': 10.5}
        Returns: [{'action_name': 'X', 'repeat': False}, ...]  (metadata stripped)
    """
    # New format: {'context': [...], 'occurrence_count': X, 'percentage': Y}
    if isinstance(pattern, dict) and 'context' in pattern:
        return pattern['context']
    
    # Old format: Just a list of action dicts
    if isinstance(pattern, list):
        return pattern
    
    # Fallback: Return empty list
    return []


def match_sequence_pattern(
    execution_history: List[str],
    required_pattern: Any  # Can be list (old) or dict (new)
) -> Tuple[bool, str]:
    """
    Match an execution history against a required leading context pattern.
    **Supports both old and new graph formats transparently.**
    """

    # ================= DEBUG OUTPUT =================
    print("History was:")
    print(list(execution_history))
    print("-" * 40)
    # ===============================================

    # Normalize the pattern to standard format (strips metadata if present)
    normalized_pattern = normalize_pattern(required_pattern)

    if not normalized_pattern:
        return True, ""

    if not execution_history and normalized_pattern:
        first_action = normalized_pattern[0].get('action_name', '')
        return False, f"History is empty, but pattern requires {first_action!r}"

    pattern_idx = 0
    history_idx = 0

    while pattern_idx < len(normalized_pattern) and history_idx < len(execution_history):
        pattern_entry = normalized_pattern[pattern_idx]
        expected_action = pattern_entry.get("action_name")
        repeat_allowed = pattern_entry.get("repeat", False)

        current_action = execution_history[history_idx]

        if current_action == expected_action:
            history_idx += 1

            # If repeat is allowed, consume all consecutive matching actions
            if repeat_allowed:
                while (
                    history_idx < len(execution_history)
                    and execution_history[history_idx] == expected_action
                ):
                    history_idx += 1

            # Move to next pattern entry
            pattern_idx += 1
        else:
            return False, (
                f"Pattern mismatch at index {history_idx}: "
                f"Expected {expected_action!r} but got {current_action!r}"
            )

    # Check if we matched the whole pattern
    if pattern_idx == len(normalized_pattern):
        return True, ""
    else:
        expected_action = normalized_pattern[pattern_idx].get("action_name")
        return False, (
            f"History ended, but pattern still required action: {expected_action!r}"
        )


def validate_workflow(
    agent_role: str,
    action_to_execute: str,
    execution_history: List[str],
    graph_yaml_path: Path
) -> Tuple[bool, Optional[str]]:
    """
    Validate if an action can be executed given execution history and control flow graph.
    
    **BACKWARD COMPATIBLE - Supports both graph formats:**
    - Old format: required_leading_contexts as list of lists
    - New format: required_leading_contexts with 'context', 'occurrence_count', 'percentage'
    
    The enforcer ignores occurrence_count and percentage fields (used by colleagues for analysis).
    
    Args:
        agent_role: The role of the agent (e.g., "AI_Assistant")
        action_to_execute: The action about to be executed (e.g., "get_channels")
        execution_history: List of actions executed so far in this run
        graph_yaml_path: Path to the agent's control flow graph YAML file
        
    Returns:
        (is_valid, error_message) tuple
        - is_valid: True if action allowed, False otherwise
        - error_message: Description of violation if is_valid=False, None otherwise
    """
    
    # Load the control flow graph
    try:
        with open(graph_yaml_path, 'r') as f:
            graph_data = yaml.safe_load(f)
    except FileNotFoundError:
        return False, f"Control flow graph not found: {graph_yaml_path}"
    except Exception as e:
        return False, f"Error loading control flow graph: {e}"
    
    # Verify agent role matches
    if graph_data.get("agent_role") != agent_role:
        graph_agent = graph_data.get('agent_role', '')
        return False, f"Agent role mismatch: expected {agent_role}, graph has {graph_agent}"
    
    # Get the node for the action we're trying to execute
    nodes = graph_data.get("nodes", {})
    if action_to_execute not in nodes:
        if f"# {action_to_execute}" in graph_data.get("nodes", {}):
             return False, f"Action {action_to_execute!r} is commented out in graph for {agent_role}"
        return False, f"Action {action_to_execute!r} not found in control flow graph for {agent_role}"
    
    action_node = nodes[action_to_execute]
    required_leading_contexts = action_node.get("required_leading_contexts", [])
    
    # If no required contexts, action is always allowed
    if not required_leading_contexts:
        return True, None
    
    # Build the proposed history (history + action about to execute)
    proposed_history = execution_history + [action_to_execute]
    
    failed_pattern_errors = []
    
    # Try to match against ANY of the required leading contexts
    # normalize_pattern() handles both old and new formats transparently
    for pattern in required_leading_contexts:
        is_match, match_error = match_sequence_pattern(proposed_history, pattern)
        if is_match:
            return True, None  # Found a matching pattern!
        else:
            failed_pattern_errors.append(match_error)
    
    # No patterns matched - violation!
    error_details = " | ".join(failed_pattern_errors)
    
    return False, (
        f"Workflow violation: Action {action_to_execute!r} attempted without required leading context. "
        f"History: {execution_history}. "
        f"Failed checks: [{error_details}]"
    )


def test_pattern_matching():
    """Test cases for both old and new formats."""
    print("=" * 60)
    print("Testing Pattern Matching (v2.2 - Both Formats)")
    print("=" * 60)
    
    # Test 1: Old format - Simple sequence
    print("\n✓ Test 1: Old format - Simple sequence")
    history = ["List files", "Read file", "Search"]
    pattern = [
        {"action_name": "List files", "repeat": False},
        {"action_name": "Read file", "repeat": False},
        {"action_name": "Search", "repeat": False}
    ]
    result, error = match_sequence_pattern(history, pattern)
    assert result == True and error == "", f"Test 1 failed: {error}"
    print("  PASSED")
    
    # Test 2: New format - Same sequence with metadata
    print("\n✓ Test 2: New format - Same sequence WITH occurrence_count/percentage")
    history = ["List files", "Read file", "Search"]
    pattern = {
        "context": [
            {"action_name": "List files", "repeat": False},
            {"action_name": "Read file", "repeat": False},
            {"action_name": "Search", "repeat": False}
        ],
        "occurrence_count": 5,
        "percentage": 10.5
    }
    result, error = match_sequence_pattern(history, pattern)
    assert result == True and error == "", f"Test 2 failed: {error}"
    print("  PASSED (metadata ignored)")
    
    # Test 3: Old format - With repeats
    print("\n✓ Test 3: Old format - Sequence with repeats")
    history = ["get_channels", "read_channel_messages", "read_channel_messages", "No_tool_used"]
    pattern = [
        {"action_name": "get_channels", "repeat": False},
        {"action_name": "read_channel_messages", "repeat": True},
        {"action_name": "No_tool_used", "repeat": False}
    ]
    result, error = match_sequence_pattern(history, pattern)
    assert result == True and error == "", f"Test 3 failed: {error}"
    print("  PASSED")
    
    # Test 4: New format - With repeats and metadata
    print("\n✓ Test 4: New format - With repeats AND metadata")
    history = ["get_channels", "read_channel_messages", "read_channel_messages", "No_tool_used"]
    pattern = {
        "context": [
            {"action_name": "get_channels", "repeat": False},
            {"action_name": "read_channel_messages", "repeat": True},
            {"action_name": "No_tool_used", "repeat": False}
        ],
        "occurrence_count": 6,
        "percentage": 8.82
    }
    result, error = match_sequence_pattern(history, pattern)
    assert result == True and error == "", f"Test 4 failed: {error}"
    print("  PASSED (metadata ignored)")
    
    # Test 5: Old format - Missing action (should fail)
    print("\n✓ Test 5: Old format - Missing action (should FAIL)")
    history = ["List files", "Search"]  # Missing "Read file"
    pattern = [
        {"action_name": "List files", "repeat": False},
        {"action_name": "Read file", "repeat": False},
        {"action_name": "Search", "repeat": False}
    ]
    result, error = match_sequence_pattern(history, pattern)
    assert result == False, "Test 5 should have failed"
    assert "Expected 'Read file' but got 'Search'" in error, f"Test 5 error message wrong: {error}"
    print(f"  PASSED (correctly detected: {error})")
    
    # Test 6: New format - Missing action (should fail)
    print("\n✓ Test 6: New format - Missing action WITH metadata (should FAIL)")
    history = ["get_channels", "No_tool_used"]  # Missing "add_user_to_channel"
    pattern = {
        "context": [
            {"action_name": "get_channels", "repeat": False},
            {"action_name": "add_user_to_channel", "repeat": False},
            {"action_name": "No_tool_used", "repeat": False}
        ],
        "occurrence_count": 3,
        "percentage": 4.41
    }
    result, error = match_sequence_pattern(history, pattern)
    assert result == False, "Test 6 should have failed"
    assert "Expected 'add_user_to_channel' but got 'No_tool_used'" in error, f"Test 6 error message wrong: {error}"
    print(f"  PASSED (correctly detected: {error})")
    
    print("\n" + "=" * 60)
    print("✅ ALL TESTS PASSED - Both formats work correctly!")
    print("=" * 60)
    print("\nThe validator will:")
    print("  • Accept old format (list of lists)")
    print("  • Accept new format (with context/occurrence_count/percentage)")
    print("  • Ignore metadata fields (occurrence_count, percentage)")
    print("  • Validate workflow patterns correctly in both cases")


if __name__ == "__main__":
    test_pattern_matching()