"""Cooperative minibatch boundaries and an injectable monotonic run clock."""
from __future__ import annotations

import threading
import time
from typing import Callable


class BudgetExpired(Exception):
    """The current unpublished cycle must be discarded."""


class ReintegrationRequired(Exception):
    """Discard the suspended private cycle before the one-time rejoin policy."""


class RunControl:
    def __init__(self, start: float, duration: float, offline: bool,
                 start_fraction: float = 0.3, end_fraction: float = 0.6,
                 clock: Callable[[], float] = time.monotonic,
                 sleep: Callable[[float], None] = time.sleep,
                 emit: Callable[..., None] = lambda *a, **kw: None) -> None:
        self.start, self.duration, self.offline = start, duration, offline
        self.leave = start_fraction * duration
        self.return_at = end_fraction * duration
        self.clock, self.sleep, self.emit = clock, sleep, emit
        self._seen: set[str] = set()
        self._lock = threading.RLock()
        self._transition_lock = threading.RLock()
        self.steps = self.examples = 0
        self.train_seconds = 0.0
        self.require_reintegration = False
        self.reintegration_handled = False
        self.reintegration_active = False
        self.reintegration_steps = self.reintegration_examples = 0
        self.reintegration_seconds = 0.0
        self.teacher_inferences = self.teacher_examples = 0
        self.teacher_seconds = 0.0
        self.teacher_model_bytes = 0

    @property
    def elapsed(self) -> float:
        return self.clock() - self.start

    @property
    def online(self) -> bool:
        t = self.elapsed
        return 0 <= t < self.duration and not (
            self.offline and self.leave <= t < self.return_at
        )

    def transitions(self) -> None:
        # Keep event snapshots ordered before a worker can publish after return.
        with self._transition_lock:
            for name, at in (("started", 0), ("offline", self.leave),
                             ("returned", self.return_at), ("budget_ended", self.duration)):
                if name in {"offline", "returned"} and not self.offline:
                    continue
                if self.elapsed >= at and name not in self._seen:
                    self._seen.add(name)
                    self.emit(name, scheduled_seconds=at)

    def checkpoint(self) -> None:
        paused = False
        while True:
            self.transitions()
            t = self.elapsed
            if t >= self.duration:
                raise BudgetExpired()
            if self.online:
                if paused:
                    self.emit("training_resumed")
                if (self.require_reintegration and self.offline and t >= self.return_at
                        and not self.reintegration_handled):
                    raise ReintegrationRequired()
                return
            if t >= 0 and not paused:
                paused = True
                self.emit("training_paused", scheduled_seconds=self.leave)
            target = self.start if t < 0 else self.start + self.return_at
            self.sleep(min(0.05, max(0.001, target - self.clock())))

    def batch_finished(self, examples: int, seconds: float) -> None:
        with self._lock:
            self.steps += 1
            self.examples += examples
            self.train_seconds += seconds
            if self.reintegration_active:
                self.reintegration_steps += 1
                self.reintegration_examples += examples
                self.reintegration_seconds += seconds

    def teacher_finished(self, examples: int, seconds: float) -> None:
        with self._lock:
            self.teacher_inferences += 1
            self.teacher_examples += examples
            self.teacher_seconds += seconds

    def counters(self) -> dict:
        with self._lock:
            return {"optimizer_steps": self.steps, "examples_processed": self.examples,
                    "train_seconds": self.train_seconds,
                    "reintegration_optimizer_steps": self.reintegration_steps,
                    "reintegration_examples": self.reintegration_examples,
                    "reintegration_train_seconds": self.reintegration_seconds,
                    "teacher_inferences": self.teacher_inferences,
                    "teacher_examples": self.teacher_examples,
                    "teacher_seconds": self.teacher_seconds,
                    "teacher_model_bytes": self.teacher_model_bytes}

    def request_timeout(self, configured: float) -> float:
        self.checkpoint()
        boundary = self.duration
        if self.offline and self.elapsed < self.leave:
            boundary = self.leave
        return max(0.001, min(configured, boundary - self.elapsed))
