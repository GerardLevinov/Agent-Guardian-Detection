"""
Policy Audit Logger
Stores comprehensive policy enforcement audit trail
"""

import sqlite3
import threading
import json
from pathlib import Path
from typing import Dict, Any
import logging

logger = logging.getLogger(__name__)


class PolicyAuditLogger:
    """
    Simplified audit logging focusing on actual data and violations.
    Stores comprehensive policy enforcement audit trail in SQLite.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize audit logger.
        
        Args:
            db_path: Path to SQLite database (will create policy_audit.db in same directory)
        """
        self.db_path = Path(db_path).parent / "policy_audit.db"
        self._lock = threading.Lock()
        self._init_audit_db()
    
    def _init_audit_db(self):
        """Initialize the simplified audit database."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS policy_audit_log (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp TEXT NOT NULL,
                            agent_role TEXT NOT NULL,
                            action_name TEXT NOT NULL,
                            input_tokens INTEGER,
                            output_tokens INTEGER,
                            processing_duration_ms INTEGER,
                            execution_time TEXT,
                            function_args TEXT,
                            policy_passed BOOLEAN,
                            violation_reasons TEXT,
                            session_id TEXT,
                            callback_instance_id TEXT
                        )
                    """)
                    conn.commit()
                    logger.info(f"Policy audit database initialized: {self.db_path}")
                    return
                    
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2
                    logger.warning(f"Audit DB init failed, retrying in {wait_time}s: {e}")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to initialize audit database: {e}")
                    raise

    def log_policy_audit(self, audit_data: Dict[str, Any]):
        """
        Log simplified policy audit information.
        
        Args:
            audit_data: Dictionary containing audit information:
                - timestamp: ISO format timestamp
                - agent_role: Name of the agent
                - action_name: Name of the action
                - input_tokens: Number of input tokens
                - output_tokens: Number of output tokens
                - processing_duration_ms: Processing time in milliseconds
                - execution_time: Execution time (HH:MM:SS)
                - function_args: Function arguments as dict
                - policy_passed: Boolean indicating if policy checks passed
                - violation_reasons: String describing violations (if any)
                - session_id: Session identifier
                - callback_instance_id: Callback instance identifier
        """
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._lock:
                    with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                        conn.execute("""
                            INSERT INTO policy_audit_log (
                                timestamp, agent_role, action_name,
                                input_tokens, output_tokens, processing_duration_ms, 
                                execution_time, function_args, policy_passed, violation_reasons,
                                session_id, callback_instance_id
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            audit_data.get('timestamp'),
                            audit_data.get('agent_role'),
                            audit_data.get('action_name'),
                            audit_data.get('input_tokens'),
                            audit_data.get('output_tokens'),
                            audit_data.get('processing_duration_ms'),
                            audit_data.get('execution_time'),
                            json.dumps(audit_data.get('function_args', {})),
                            audit_data.get('policy_passed'),
                            audit_data.get('violation_reasons'),
                            audit_data.get('session_id'),
                            audit_data.get('callback_instance_id')
                        ))
                        conn.commit()
                        
                        status = "PASSED" if audit_data.get('policy_passed') else "FAILED"
                        logger.info(
                            f"Audit logged: {audit_data.get('agent_role')} -> "
                            f"{audit_data.get('action_name')} -> {status}"
                        )
                        return
                        
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 0.5
                    logger.debug(f"Audit DB locked, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to log policy audit: {e}")
                    break
            except Exception as e:
                logger.error(f"Failed to log policy audit: {e}")
                break