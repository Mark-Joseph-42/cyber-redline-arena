"""
Lightweight infrastructure simulation for Cyber-Redline Arena.

This module intentionally supports two modes:
1) dry-run (default): no external side effects, only simulated pod state changes
2) local shell verification: optional command execution for exploit verification
"""

from __future__ import annotations

import os
import subprocess
from typing import Dict, Any


class PodStateManager:
    """Tiny pod/container state manager inspired by Kube SRE style sandboxes."""

    def __init__(self) -> None:
        self.mode = os.getenv("CYBER_POD_MODE", "dry-run")
        self.node_status: Dict[str, Dict[str, Any]] = {}

    def reset(self, node_ids):
        self.node_status = {
            nid: {
                "pod_state": "Healthy",
                "compromised": False,
                "last_command": None,
                "last_exit_code": 0,
            }
            for nid in node_ids
        }

    def apply_compromise(self, node_id: str, exploit_command: str = "") -> Dict[str, Any]:
        if node_id not in self.node_status:
            return {"ok": False, "msg": f"Node {node_id} not registered in pod manager."}

        result = {
            "ok": True,
            "node_id": node_id,
            "mode": self.mode,
            "verified": True,
            "exit_code": 0,
            "stdout": "",
            "stderr": "",
        }

        if self.mode == "shell-verify" and exploit_command:
            try:
                proc = subprocess.run(
                    exploit_command,
                    shell=True,
                    capture_output=True,
                    text=True,
                    timeout=5,
                )
                result["exit_code"] = proc.returncode
                result["stdout"] = (proc.stdout or "").strip()
                result["stderr"] = (proc.stderr or "").strip()
                result["verified"] = proc.returncode == 0
            except Exception as exc:
                result["ok"] = False
                result["verified"] = False
                result["stderr"] = str(exc)
                result["exit_code"] = 1

        self.node_status[node_id]["pod_state"] = "Compromised"
        self.node_status[node_id]["compromised"] = True
        self.node_status[node_id]["last_command"] = exploit_command or "simulated-exploit"
        self.node_status[node_id]["last_exit_code"] = int(result["exit_code"])
        return result

    def status_snapshot(self) -> Dict[str, Dict[str, Any]]:
        return dict(self.node_status)

