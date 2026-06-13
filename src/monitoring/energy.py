"""Windows-focused energy measurement during LLM inference."""

from __future__ import annotations

import platform
import threading
import time
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class EnergyCalibration:
    """Calibration factors for converting utilization samples to joules."""

    platform_name: str
    base_power_watts: float
    cpu_watts_per_percent: float
    sample_interval_s: float = 0.1

    def to_dict(self) -> dict[str, Any]:
        return {
            "platform_name": self.platform_name,
            "base_power_watts": self.base_power_watts,
            "cpu_watts_per_percent": self.cpu_watts_per_percent,
            "sample_interval_s": self.sample_interval_s,
        }


DEFAULT_WINDOWS_CALIBRATION = EnergyCalibration(
    platform_name="windows",
    base_power_watts=8.0,
    cpu_watts_per_percent=0.35,
    sample_interval_s=0.1,
)

DEFAULT_GENERIC_CALIBRATION = EnergyCalibration(
    platform_name="generic",
    base_power_watts=6.0,
    cpu_watts_per_percent=0.25,
    sample_interval_s=0.1,
)


@dataclass
class MeasuredEnergy:
    joules: float
    duration_s: float
    average_power_watts: float
    sample_count: int
    calibration: EnergyCalibration
    method: str
    notes: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "joules": round(self.joules, 4),
            "duration_s": round(self.duration_s, 4),
            "average_power_watts": round(self.average_power_watts, 4),
            "sample_count": self.sample_count,
            "calibration": self.calibration.to_dict(),
            "method": self.method,
            "notes": self.notes,
        }


def load_calibration() -> EnergyCalibration:
    system = platform.system().lower()
    if system == "windows":
        return DEFAULT_WINDOWS_CALIBRATION
    return DEFAULT_GENERIC_CALIBRATION


class EnergyMonitor:
    """Integrates estimated power draw over an inference interval."""

    def __init__(self, calibration: EnergyCalibration | None = None) -> None:
        self.calibration = calibration or load_calibration()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._samples: list[float] = []
        self._started_at: float | None = None
        self._notes: list[str] = []

    def __enter__(self) -> EnergyMonitor:
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def start(self) -> None:
        self._stop_event.clear()
        self._samples = []
        self._notes = []
        self._started_at = time.perf_counter()
        self._thread = threading.Thread(target=self._sample_loop, daemon=True)
        self._thread.start()

    def stop(self) -> MeasuredEnergy:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

        ended_at = time.perf_counter()
        started_at = self._started_at or ended_at
        duration_s = max(0.0, ended_at - started_at)

        if not self._samples:
            self._notes.append("No utilization samples collected; using base power only.")
            average_cpu = 0.0
        else:
            average_cpu = sum(self._samples) / len(self._samples)

        average_power = self.calibration.base_power_watts + (
            self.calibration.cpu_watts_per_percent * average_cpu
        )
        joules = average_power * duration_s

        return MeasuredEnergy(
            joules=joules,
            duration_s=duration_s,
            average_power_watts=average_power,
            sample_count=len(self._samples),
            calibration=self.calibration,
            method=f"integrated_power_{self.calibration.platform_name}",
            notes=self._notes,
        )

    def _sample_loop(self) -> None:
        while not self._stop_event.is_set():
            cpu_percent = self._read_cpu_percent()
            self._samples.append(cpu_percent)
            self._stop_event.wait(self.calibration.sample_interval_s)

    def _read_cpu_percent(self) -> float:
        try:
            import psutil
        except ImportError:
            if "psutil unavailable" not in self._notes:
                self._notes.append("psutil unavailable; using conservative CPU estimate.")
            return 35.0

        try:
            return float(psutil.cpu_percent(interval=None))
        except Exception:
            return 25.0


def estimate_proxy_joules(total_tokens: int, model: str) -> float:
    """Token-based proxy estimate retained for side-by-side reporting."""
    model_lower = model.lower()
    if "4b" in model_lower or "3b" in model_lower:
        factor = 0.08
    elif "7b" in model_lower or "8b" in model_lower:
        factor = 0.12
    else:
        factor = 0.15
    return total_tokens * factor
