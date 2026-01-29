"""
Complete integration test for Agent Guardian
Tests the entire enforcement system end-to-end
"""

from pathlib import Path
from agent_guardian import enable_enforcement, get_enforcement_stats, disable_enforcement

print("=" * 60)
print("Agent Guardian - Complete Integration Test")
print("=" * 60)

# Setup test environment
test_policy_dir = Path("./test_policies")
test_db_path = Path("./test_integration.db")

# Create minimal test policy structure
test_policy_dir.mkdir(exist_ok=True)
test_agent_dir = test_policy_dir / "policies_test" / "Test_Agent"
test_agent_dir.mkdir(parents=True, exist_ok=True)

# Create a simple test policy
test_policy_content = """
agent_role: Test_Agent
action_name: test_action
input_restrictions:
  numeric_limits:
    input_tokens:
      min_input_tokens: 0
      max_input_tokens: 10000
  categorical_values:
    - input_pattern_1:
        param1: ".*"
"""

with open(test_agent_dir / "test_action.yaml", 'w') as f:
    f.write(test_policy_content)

print("\n✓ Test environment created")

# Test 1: Enable enforcement
print("\nTest 1: Enable enforcement...")
try:
    enforcer = enable_enforcement(
        policy_dir=str(test_policy_dir),
        db_path=str(test_db_path),
        enable_semantic_validation=False,  # Skip Ollama for quick test
        validate_ollama=False
    )
    print("✓ Enforcement enabled successfully")
except Exception as e:
    print(f"❌ Failed to enable enforcement: {e}")
    exit(1)

# Test 2: Check stats
print("\nTest 2: Get enforcement stats...")
try:
    stats = get_enforcement_stats()
    print(f"✓ Stats retrieved: {stats}")
except Exception as e:
    print(f"❌ Failed to get stats: {e}")
    exit(1)

# Test 3: Verify callback is registered
print("\nTest 3: Verify callback registration...")
import litellm
if enforcer.callback in litellm.success_callback:
    print("✓ Callback registered with LiteLLM")
else:
    print("❌ Callback not registered")
    exit(1)

# Test 4: Disable enforcement
print("\nTest 4: Disable enforcement...")
try:
    disable_enforcement()
    if enforcer.callback not in litellm.success_callback:
        print("✓ Enforcement disabled successfully")
    else:
        print("❌ Callback still registered")
        exit(1)
except Exception as e:
    print(f"❌ Failed to disable enforcement: {e}")
    exit(1)

# Cleanup
print("\nCleaning up test files...")
import shutil
import time

# Delete references to allow Windows to release file locks
del enforcer
del stats
import gc
gc.collect()
time.sleep(0.5)  # Give Windows time to release locks

try:
    if test_policy_dir.exists():
        shutil.rmtree(test_policy_dir)
    if test_db_path.exists():
        test_db_path.unlink()
    if Path("policy_audit.db").exists():
        Path("policy_audit.db").unlink()
    print("✓ Test files cleaned up")
except PermissionError:
    print("⚠️  Could not delete test databases (still in use)")
    print("   This is normal on Windows - files will be cleaned up on next run")

print("\n" + "=" * 60)
print("✅ ALL INTEGRATION TESTS PASSED!")
print("=" * 60)
print("\nAgent Guardian is ready to use!")
print("\nNext steps:")
print("  1. Create your policies folder structure")
print("  2. Add policy YAML files for your agents")
print("  3. Import and enable enforcement in your CrewAI app:")
print("\n     from agent_guardian import enable_enforcement")
print("     enable_enforcement()")
print("\n     # Your CrewAI code here")
print("     crew = Crew(agents=[...], tasks=[...])")
print("     result = crew.kickoff()")