"""
The Kairos Engine - Cognitive Autopilot Core Brain Interface

Provides convenience wrappers around KairosEngine for running
complete missions, checking status, and graceful shutdown.
"""

import logging
from typing import List, Dict, Any, Optional, Union

from src.core.engine import KairosEngine
from src.core.telemetry import TelemetryReport
from src.core.exceptions import KairosError, ModelLoadError

logger = logging.getLogger("kairos.brain")


class GemmaBrain(KairosEngine):
    """
    GemmaBrain wrapper class maintaining full backwards compatibility
    with the Kairos Master Engine Orchestrator.
    Adds run_mission() convenience method for single-call operation.
    """

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = "./models/gemma-4-E2B-it-Q4_K_M.gguf"
        super().__init__(model_path=model_path)

    def run_mission(self, mission_request: str,
                    telemetry: Optional[Union[Dict[str, Any], TelemetryReport]] = None,
                    drone_id: str = "KAIROS-01") -> Dict[str, Any]:
        """
        Run a complete mission cycle: pre-flight plan → in-flight decision → report.

        Args:
            mission_request: The mission description string.
            telemetry: Optional in-flight telemetry data for Phase B.
            drone_id: Drone identifier.

        Returns:
            Dict with plan, decision (if telemetry provided), and summary.
        """
        result = {"drone_id": drone_id}

        # Phase A: Pre-flight planning
        try:
            plan = self.pre_flight_plan(mission_request, drone_id=drone_id)
            result["plan"] = plan
        except KairosError as exc:
            logger.error(f"Pre-flight planning failed: {exc}")
            result["plan_error"] = str(exc)
            return result

        # Phase B: In-flight decision (if telemetry provided)
        if telemetry is not None:
            try:
                decision = self.in_flight_decision(telemetry, mission_context=plan)
                result["decision"] = decision
            except KairosError as exc:
                logger.error(f"In-flight decision failed: {exc}")
                result["decision_error"] = str(exc)

        # Post-flight report
        try:
            result["summary"] = self.post_flight_report()
        except Exception as exc:
            logger.error(f"Report generation failed: {exc}")

        return result


class KairosBrain(KairosEngine):
    """
    Primary brain interface with status monitoring and lifecycle management.
    """

    _active = False

    def __init__(self, model_path: str = None):
        if model_path is None:
            model_path = "./models/gemma-4-E2B-it-Q4_K_M.gguf"
        super().__init__(model_path=model_path)
        self._active = True

    def get_status(self) -> Dict[str, Any]:
        """Return the current engine status including health check."""
        status = super().get_status()
        status["active"] = self._active
        status["brain_class"] = self.__class__.__name__
        return status

    def shutdown(self):
        """Gracefully shut down the engine, flushing logs and releasing resources."""
        if not self._active:
            return

        logger.info("Kairos Brain shutting down...")
        from src.utils.logger import KairosLogger
        KairosLogger.audit("engine_shutdown", self.get_status())

        self._active = False
        logger.info("Kairos Brain shutdown complete.")

    def is_active(self) -> bool:
        """Check if the engine is currently active."""
        return self._active
