"""Background worker thread: takes due jobs from the store one at a time."""

from __future__ import annotations

import logging
import threading

from smartep.api.pipeline import JobRunner

log = logging.getLogger("smartep.worker")


class Worker:
    def __init__(self, runner: JobRunner, poll_interval_s: float):
        self.runner, self.poll = runner, poll_interval_s
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._loop, name="smartep-worker", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout)

    def run_once(self) -> bool:
        """Process one due job; returns False when nothing was due."""
        job = self.runner.store.claim_next()
        if job is None:
            return False
        self.runner.run(job)
        return True

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                if not self.run_once():
                    self._stop.wait(self.poll)
            except Exception:  # never let the thread die
                log.exception("worker loop error")
                self._stop.wait(self.poll)
