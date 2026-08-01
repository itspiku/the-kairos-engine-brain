"""
The Kairos Engine — Tool Executor & Implementations Test Suite
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.tools.executor import ToolExecutor
from src.tools.definitions import get_tool_definitions
from src.core.exceptions import ToolExecutionError


class TestToolExecutor(unittest.TestCase):

    def setUp(self):
        self.executor = ToolExecutor()

    def test_available_tools_not_empty(self):
        tools = self.executor.get_available_tools()
        self.assertGreater(len(tools), 5)

    def test_all_definitions_have_implementations(self):
        definitions = get_tool_definitions()
        available = self.executor.get_available_tools()
        for tool_def in definitions:
            name = tool_def["function"]["name"]
            self.assertIn(name, available, f"Tool '{name}' has definition but no implementation")

    def test_delivery_priority_oxytocin_critical(self):
        result = self.executor.execute("get_delivery_priority", {"payload_type": "oxytocin"})
        self.assertEqual(result, "critical")

    def test_delivery_priority_blood_critical(self):
        result = self.executor.execute("get_delivery_priority", {"payload_type": "blood plasma"})
        self.assertEqual(result, "critical")

    def test_delivery_priority_bandages_standard(self):
        result = self.executor.execute("get_delivery_priority", {"payload_type": "bandages"})
        self.assertEqual(result, "standard")

    def test_delivery_priority_vitamins_standard(self):
        result = self.executor.execute("get_delivery_priority", {"payload_type": "vitamins"})
        self.assertEqual(result, "standard")

    def test_unknown_tool_raises_error(self):
        with self.assertRaises(ToolExecutionError):
            self.executor.execute("nonexistent_tool_xyz", {})

    def test_get_battery_state_returns_dict(self):
        result = self.executor.execute("get_battery_state", {"drone_id": "KAIROS-01"})
        self.assertIsInstance(result, dict)
        self.assertIn("soc", result)
        self.assertIn("voltage", result)

    def test_battery_soc_in_valid_range(self):
        result = self.executor.execute("get_battery_state", {"drone_id": "TEST-01"})
        self.assertGreaterEqual(result["soc"], 0)
        self.assertLessEqual(result["soc"], 100)

    def test_find_lzs_returns_list(self):
        result = self.executor.execute(
            "find_contingency_landing_zones",
            {"current_pos": [28.35, 83.88], "radius_m": 5000}
        )
        self.assertIsInstance(result, list)

    def test_find_lzs_each_has_required_keys(self):
        result = self.executor.execute(
            "find_contingency_landing_zones",
            {"current_pos": [28.35, 83.88], "radius_m": 50000}
        )
        for lz in result:
            self.assertIn("id", lz)
            self.assertIn("pos", lz)

    def test_airspace_check_returns_dict(self):
        result = self.executor.execute(
            "check_airspace_restrictions",
            {"bbox": {"north": 29, "south": 28, "east": 84, "west": 83}}
        )
        self.assertIsInstance(result, dict)
        self.assertIn("restricted", result)

    def test_log_decision_returns_logged_true(self):
        result = self.executor.execute(
            "log_decision",
            {"decision": "DIVERT", "rationale": "Test rationale"}
        )
        self.assertIsInstance(result, dict)
        self.assertTrue(result.get("logged", False))

    def test_execution_log_populated(self):
        self.executor.execute("get_delivery_priority", {"payload_type": "insulin"})
        log = self.executor.get_execution_log()
        self.assertGreater(len(log), 0)

    def test_batch_execute(self):
        calls = [
            {"name": "get_delivery_priority", "arguments": {"payload_type": "oxytocin"}},
            {"name": "get_delivery_priority", "arguments": {"payload_type": "bandages"}},
        ]
        results = self.executor.execute_batch(calls)
        self.assertEqual(len(results), 2)
        self.assertEqual(results[0]["status"], "success")


if __name__ == "__main__":
    unittest.main()
