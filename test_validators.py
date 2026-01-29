"""Test core validator modules"""
from pathlib import Path
from agent_guardian.core import validate_workflow, match_sequence_pattern

print("Test 1: Workflow Validator - Pattern Matching...")

# Test simple pattern match
history = ["action1", "action2", "action3"]
pattern = [
    {"action_name": "action1", "repeat": False},
    {"action_name": "action2", "repeat": False},
    {"action_name": "action3", "repeat": False}
]

is_match, error = match_sequence_pattern(history, pattern)
assert is_match == True, f"Pattern should match: {error}"
print("✓ Simple pattern matching works")

# Test pattern with metadata (new format)
pattern_with_metadata = {
    "context": [
        {"action_name": "action1", "repeat": False},
        {"action_name": "action2", "repeat": False}
    ],
    "occurrence_count": 5,
    "percentage": 10.0
}

history2 = ["action1", "action2"]
is_match, error = match_sequence_pattern(history2, pattern_with_metadata)
assert is_match == True, f"Pattern with metadata should match: {error}"
print("✓ Pattern with metadata works (metadata ignored)")

# Test pattern mismatch
history3 = ["action1", "action3"]  # Missing action2
pattern3 = [
    {"action_name": "action1", "repeat": False},
    {"action_name": "action2", "repeat": False},
    {"action_name": "action3", "repeat": False}
]

is_match, error = match_sequence_pattern(history3, pattern3)
assert is_match == False, "Pattern should NOT match"
print(f"✓ Pattern mismatch detected correctly: {error}")

print("\n✅ All workflow validator tests passed!")

print("\nTest 2: Semantic Validator (skipped - requires Ollama)")
print("  To test semantic validator:")
print("  1. Install Ollama: https://ollama.com/download")
print("  2. Run: ollama pull qwen2.5:7b-instruct")
print("  3. Start: ollama serve")
print("  4. Uncomment test code below")

# Uncomment to test semantic validator (requires Ollama running)
from agent_guardian.core import get_semantic_validator

validator = get_semantic_validator(
    ollama_host="http://localhost:11434",
    model="qwen2.5:7b-instruct"
)

result = validator.validate_field(
    field_name="message",
    field_value="Hello, how are you?",
    semantic_description="Should be a normal greeting message"
)

print(f"Semantic validation result: {result}")
assert result['is_violation'] == False, "Normal message should pass"
print("✓ Semantic validation works")

print("\n✅ Core validators ready for integration!")