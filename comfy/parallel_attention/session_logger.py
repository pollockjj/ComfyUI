"""Session-based logging for Parallel Attention workflows.

Writes all ⚡ [Parallel-Attention] logs to a per-session markdown file
for clean, unpolluted log analysis.

Pattern: Singleton logger that captures parallel-attention events during workflow execution.
"""

import logging
from pathlib import Path
from datetime import datetime


class SessionLogger:
    """Singleton logger for parallel-attention session tracking."""
    
    _instance = None
    
    def __init__(self):
        self.session_id = None
        self.output_dir = None
        self.log_file = None
        self.started_at = None
        self.messages = []
    
    @classmethod
    def get_instance(cls):
        """Get singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def start_session(self, session_id: str = None):
        """Start a new logging session."""
        # Generate session ID if not provided
        if session_id is None:
            from datetime import datetime
            session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        self.session_id = session_id
        self.started_at = datetime.now()
        self.messages = []
        logging.debug(f"⚡ [SessionLogger] Started session: {session_id}")
    
    def log(self, message: str):
        """Log a message (buffered until finalization)."""
        if self.session_id:
            timestamp = datetime.now()
            self.messages.append((timestamp, message))
    
    def is_active(self):
        """Check if a session is active."""
        return self.session_id is not None
    
    def finalize_session(self, output_dir: str):
        """Write buffered logs to markdown file in output directory."""
        if not self.session_id:
            return
        
        try:
            # Create output directory
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            
            # Copy full ComfyUI log to session directory
            comfyui_log = Path("/home/johnj/ComfyUI/user/comfyui.log")
            if comfyui_log.exists():
                import shutil
                full_log_dest = output_path / "comfyui_full.log"
                shutil.copy2(comfyui_log, full_log_dest)
                logging.info(f"⚡ [SessionLogger] Copied full log: {full_log_dest}")
            
            # Write markdown file
            log_file = output_path / "parallel_attention_session.md"
            
            with open(log_file, "w") as f:
                # Header
                f.write(f"# Parallel Attention Session Log\n\n")
                f.write(f"**Session ID:** `{self.session_id}`  \n")
                f.write(f"**Started:** {self.started_at.strftime('%Y-%m-%d %H:%M:%S')}  \n\n")
                f.write(f"---\n\n")
                
                # Messages
                for timestamp, message in self.messages:
                    ts_str = timestamp.strftime("%H:%M:%S.%f")[:-3]
                    f.write(f"`{ts_str}` {message}\n")
                
                # Footer
                ended_at = datetime.now()
                f.write(f"\n---\n\n")
                f.write(f"**Ended:** {ended_at.strftime('%Y-%m-%d %H:%M:%S')}\n")
            
            logging.info(f"⚡ [SessionLogger] Wrote session log: {log_file}")
        except Exception as e:
            logging.error(f"⚡ [SessionLogger] Failed to write session log: {e}")
        finally:
            # Clear session even if write fails
            self.session_id = None
            self.output_dir = None
            self.started_at = None
            self.messages = []
