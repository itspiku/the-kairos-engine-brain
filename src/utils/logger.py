"""
The Kairos Engine - High-Contrast CLI Dashboard Logger
"""

import sys
import time

class KairosLogger:
    @staticmethod
    def header(title: str):
        print("\n" + "=" * 64)
        print(f"  ⚡ THE KAIROS ENGINE | {title.upper()}")
        print("=" * 64)

    @staticmethod
    def info(msg: str):
        print(f"   [INFO] {msg}")

    @staticmethod
    def ok(msg: str):
        print(f"   [OK] {msg}")

    @staticmethod
    def warn(msg: str):
        print(f"   [WARN] {msg}")

    @staticmethod
    def error(msg: str):
        print(f"   [ERROR] {msg}")

    @staticmethod
    def decision(decision: str, meets_sla: bool = True):
        sla_tag = "[SLA PASS <2s]" if meets_sla else "[SLA WARN >2s]"
        print(f"\n   🎯 KAIROS DECISION: {decision}")
        print(f"   ⚡ LATENCY STATUS : {sla_tag}\n")
