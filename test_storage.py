"""Test storage layer modules"""
from pathlib import Path
from agent_guardian.storage import SQLiteLogger, PolicyAuditLogger

# Setup test database paths
db_path = Path("./test_storage.db")
audit_path = Path("./test_audit.db")

print("Test 1: SQLite Logger...")
logger = SQLiteLogger(db_path)
print(f"✓ SQLite logger created: {db_path}")

# Log a test activity
logger.log_activity(
    session_id="test_session_1",
    agent_role="Test_Agent",
    action_name="test_action",
    function_args={"param1": "value1"},
    allowed_next_moves=["action2", "action3"]
)
print("✓ Activity logged")

# Get statistics
stats = logger.get_activity_stats()
print(f"✓ Stats: {stats}")

# Get recent activities
activities = logger.get_recent_activities(limit=5)
print(f"✓ Recent activities: {len(activities)} found")

print("\nTest 2: Policy Audit Logger...")
audit_logger = PolicyAuditLogger(audit_path)
print(f"✓ Audit logger created: {audit_logger.db_path}")

# Log a test audit
from datetime import datetime
audit_logger.log_policy_audit({
    'timestamp': datetime.now().isoformat()[:19],
    'agent_role': 'Test_Agent',
    'action_name': 'test_action',
    'input_tokens': 100,
    'output_tokens': 50,
    'processing_duration_ms': 250,
    'execution_time': '10:30:00',
    'function_args': {'param1': 'value1'},
    'policy_passed': True,
    'violation_reasons': '',
    'session_id': 'test_session_1',
    'callback_instance_id': 'test_callback_1'
})
print("✓ Audit logged")

print("\n✅ All storage tests passed!")
print("\nNote: Test databases (test_storage.db, policy_audit.db) left for inspection")
print("      You can delete them manually or they'll be overwritten on next test run")
