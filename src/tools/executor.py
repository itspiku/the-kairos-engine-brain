"""
Tool Executor — Routes tool calls to real implementations with mock fallback.

Provides execution timing, structured error handling, and batch execution.
"""

import time
import logging
from typing import Dict, Any, List

from src.core.exceptions import ToolExecutionError

logger = logging.getLogger("kairos.executor")


class ToolExecutor:
    """Routes cognitive tool calls to Python implementations."""

    def __init__(self):
        self.tools_map = self._build_tools_map()
        self._execution_log: List[Dict] = []

    def _build_tools_map(self) -> Dict[str, callable]:
        """Build tool routing map: real implementations first, mocks as fallback."""
        tools = {}

        # Load real implementations
        try:
            from src.tools import implementations as impl
            tools.update({
                "get_wind_forecast": impl.get_wind_forecast,
                "get_terrain_elevation_profile": impl.get_terrain_elevation_profile,
                "get_battery_state": impl.get_battery_state,
                "compute_energy_aware_route": impl.compute_energy_aware_route,
                "find_contingency_landing_zones": impl.find_contingency_landing_zones,
                "check_airspace_restrictions": impl.check_airspace_restrictions,
                "get_delivery_priority": impl.get_delivery_priority,
                "recompute_route_from_current_pos": impl.recompute_route_from_current_pos,
                "select_best_landing_zone": impl.select_best_landing_zone,
                "log_decision": impl.log_decision,
                "assess_risk": impl.assess_risk,
            })
            logger.info("Loaded real tool implementations")
        except ImportError:
            logger.warning("Real implementations unavailable, falling back to mocks")
            try:
                from src.tools import mocks
                tools.update({
                    "get_wind_forecast": mocks.get_wind_forecast,
                    "get_terrain_elevation_profile": mocks.get_terrain_elevation_profile,
                    "get_battery_state": mocks.get_battery_state,
                    "compute_energy_aware_route": mocks.compute_energy_aware_route,
                    "find_contingency_landing_zones": mocks.find_contingency_landing_zones,
                    "check_airspace_restrictions": mocks.check_airspace_restrictions,
                    "get_delivery_priority": mocks.get_delivery_priority,
                    "recompute_route_from_current_pos": mocks.recompute_route_from_current_pos,
                    "select_best_landing_zone": mocks.select_best_landing_zone,
                    "log_decision": mocks.log_decision,
                    "assess_risk": getattr(mocks, "assess_risk", None),
                })
            except ImportError:
                logger.error("Neither implementations nor mocks available")

        return tools

    def execute(self, name: str, arguments: Dict = None) -> Any:
        """
        Execute a tool by name with timing and error handling.
        Returns the tool result or an error dict.
        """
        if arguments is None:
            arguments = {}

        func = self.tools_map.get(name)
        if func is None:
            raise ToolExecutionError(name, reason="Tool not found")

        start = time.time()
        try:
            result = func(**arguments)
            elapsed = round(time.time() - start, 4)

            self._execution_log.append({
                "tool": name,
                "elapsed_sec": elapsed,
                "status": "success",
            })
            logger.debug(f"Tool '{name}' executed in {elapsed}s")
            return result

        except TypeError as exc:
            # Argument mismatch — try with positional args
            raise ToolExecutionError(name, reason=f"Argument error: {exc}", arguments=arguments) from exc
        except ToolExecutionError:
            raise
        except Exception as exc:
            elapsed = round(time.time() - start, 4)
            self._execution_log.append({
                "tool": name,
                "elapsed_sec": elapsed,
                "status": "error",
                "error": str(exc),
            })
            raise ToolExecutionError(name, reason=str(exc), arguments=arguments) from exc

    def execute_batch(self, calls: List[Dict]) -> List[Dict]:
        """Execute multiple tool calls and return results."""
        results = []
        for call in calls:
            name = call.get("name", "")
            args = call.get("arguments", {})
            try:
                result = self.execute(name, args)
                results.append({"name": name, "result": result, "status": "success"})
            except ToolExecutionError as exc:
                results.append({"name": name, "error": str(exc), "status": "error"})
        return results

    def get_available_tools(self) -> List[str]:
        """Return list of all registered tool names."""
        return [name for name, func in self.tools_map.items() if func is not None]

    def get_execution_log(self) -> List[Dict]:
        """Return the execution log for performance analysis."""
        return list(self._execution_log)
