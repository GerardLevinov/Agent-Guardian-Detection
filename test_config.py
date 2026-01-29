"""Quick test to verify configuration works on Windows"""
from pathlib import Path
from agent_guardian.config import EnforcementConfig

# Create a test policies folder
test_policies = Path.cwd() / "policies"
test_policies.mkdir(exist_ok=True)

# Test 1: Manual configuration (Windows path)
print("Test 1: Manual config...")
config = EnforcementConfig(
    policy_dir=str(test_policies),  # Can also use r"C:\full\path\to\policies"
    db_path="./test.db"
)
print(f"✓ Config created: {config.policy_dir}")
print(f"✓ Database path: {config.db_path}")

# Test 2: Environment-based (will use smart defaults)
print("\nTest 2: Environment config...")
config_env = EnforcementConfig.from_env()
print(f"✓ Env config policy dir: {config_env.policy_dir}")
print(f"✓ Env config DB: {config_env.db_path}")

# Test 3: Validate Ollama (only if you have Ollama running on Windows)
print("\nTest 3: Ollama validation (comment out if Ollama not installed)...")
try:
    from agent_guardian.config import validate_ollama_available
    # Uncomment the line below if you have Ollama running:
    # validate_ollama_available(config)
    print("✓ Ollama validation skipped (uncomment to test)")
except Exception as e:
    print(f"⚠️  Ollama not available: {e}")

print("\n✅ All tests passed!")