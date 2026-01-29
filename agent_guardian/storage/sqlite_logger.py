"""
SQLite Logger for Agent Activities
Extracted from policy_callback.py for reusability
"""

import sqlite3
import threading
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, List, Optional
import logging

logger = logging.getLogger(__name__)


class SQLiteLogger:
    """
    Thread-safe SQLite logger for agent activities.
    Works in both Docker and Windows environments.
    """
    
    def __init__(self, db_path: Path):
        """
        Initialize SQLite logger.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self._lock = threading.Lock()
        self._ensure_directory()
        self._init_db()
        
    def _ensure_directory(self):
        """Ensure the database directory exists with proper permissions."""
        try:
            self.db_path.parent.mkdir(parents=True, exist_ok=True)
            logger.debug(f"Database directory ensured: {self.db_path.parent}")
        except Exception as e:
            logger.warning(f"Could not create database directory: {e}")
    
    def _init_db(self):
        """Initialize the SQLite database and create tables if they don't exist."""
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                    conn.execute("""
                        CREATE TABLE IF NOT EXISTS agent_activities (
                            id INTEGER PRIMARY KEY AUTOINCREMENT,
                            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                            session_id TEXT,
                            agent_role TEXT NOT NULL,
                            action_name TEXT NOT NULL,
                            function_args TEXT,
                            allowed_next_moves TEXT
                        )
                    """)
                    conn.commit()
                    logger.info(f"SQLite database initialized: {self.db_path}")
                    return
                    
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2
                    logger.warning(
                        f"Database locked, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Database operation failed: {e}")
                    raise
            except Exception as e:
                if attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 2
                    logger.warning(
                        f"Init failed, retrying in {wait_time}s "
                        f"(attempt {attempt + 1}/{max_retries}): {e}"
                    )
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to initialize database after {max_retries} attempts: {e}")
                    raise

    def log_activity(
        self, 
        session_id: str, 
        agent_role: str, 
        action_name: str, 
        function_args: Optional[Dict[str, Any]] = None, 
        allowed_next_moves: Optional[List[str]] = None
    ):
        """
        Log an agent activity with retry mechanism.
        
        Args:
            session_id: Unique session identifier
            agent_role: Name of the agent role
            action_name: Name of the action being executed
            function_args: Arguments passed to the function
            allowed_next_moves: List of allowed next actions
        """
        if not agent_role or not action_name:
            logger.debug("Skipping activity log: missing agent_role or action_name")
            return
        
        max_retries = 3
        for attempt in range(max_retries):
            try:
                with self._lock:
                    with sqlite3.connect(str(self.db_path), timeout=30.0) as conn:
                        args_json = json.dumps(function_args) if function_args else "{}"
                        moves_json = json.dumps(allowed_next_moves) if allowed_next_moves else "[]"
                        
                        conn.execute("""
                            INSERT INTO agent_activities 
                            (timestamp, session_id, agent_role, action_name, function_args, allowed_next_moves)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """, (
                            datetime.now().isoformat()[:19],
                            session_id,
                            agent_role,
                            action_name,
                            args_json,
                            moves_json
                        ))
                        conn.commit()
                        logger.info(f"Activity logged: {agent_role} -> {action_name}")
                        return
                        
            except sqlite3.OperationalError as e:
                if "database is locked" in str(e).lower() and attempt < max_retries - 1:
                    import time
                    wait_time = (attempt + 1) * 0.5
                    logger.debug(f"Database locked during log_activity, retrying in {wait_time}s")
                    time.sleep(wait_time)
                    continue
                else:
                    logger.error(f"Failed to log activity: {e}")
                    break
            except Exception as e:
                logger.error(f"Unexpected error logging activity: {e}")
                break
    
    def get_recent_activities(self, limit: Optional[int] = None) -> List[Dict[str, Any]]:
        """
        Get recent agent activities from the database.
        
        Args:
            limit: Maximum number of activities to return
        
        Returns:
            List of activity dictionaries
        """
        try:
            with sqlite3.connect(str(self.db_path), timeout=15.0) as conn:
                conn.row_factory = sqlite3.Row
                
                query = "SELECT * FROM agent_activities ORDER BY timestamp DESC"
                if limit is not None:
                    query += f" LIMIT {limit}"
                
                cursor = conn.execute(query)
                activities = [dict(row) for row in cursor.fetchall()]
                
                return activities
                
        except Exception as e:
            logger.error(f"Error fetching recent activities: {e}")
            return []
    
    def get_activity_stats(self) -> Dict[str, Any]:
        """
        Get summary statistics about logged activities.
        
        Returns:
            Dictionary with statistics (total_activities, unique_roles, unique_actions)
        """
        try:
            with sqlite3.connect(str(self.db_path), timeout=15.0) as conn:
                cursor = conn.execute("""
                    SELECT 
                        COUNT(*) as total_activities,
                        COUNT(DISTINCT agent_role) as unique_roles,
                        COUNT(DISTINCT action_name) as unique_actions
                    FROM agent_activities
                """)
                stats = cursor.fetchone()
                
                return {
                    "total_activities": stats[0] or 0,
                    "unique_roles": stats[1] or 0,
                    "unique_actions": stats[2] or 0
                }
        except Exception as e:
            logger.error(f"Error getting activity stats: {e}")
            return {
                "total_activities": 0,
                "unique_roles": 0,
                "unique_actions": 0,
                "error": str(e)
            }

    def get_execution_history(self, session_id: str) -> List[str]:
        """
        Get the execution history (list of action names) for a specific session.
        Returns actions in chronological order (oldest first).
        
        Args:
            session_id: Session identifier
        
        Returns:
            List of action names in chronological order
        """
        try:
            with self._lock:
                with sqlite3.connect(str(self.db_path), timeout=15.0) as conn:
                    cursor = conn.execute("""
                        SELECT action_name 
                        FROM agent_activities 
                        WHERE session_id = ?
                        ORDER BY timestamp ASC
                    """, (session_id,))
                    
                    actions = [row[0] for row in cursor.fetchall()]
                    logger.debug(f"Execution history for {session_id} (from DB): {actions}")
                    return actions
                    
        except Exception as e:
            logger.error(f"Error fetching execution history: {e}")
            return []