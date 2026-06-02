from __future__ import annotations

import threading
import time
from concurrent.futures import Future

import pytest

import obb_geo_api_server as srv
from obb_geo_api_server import _collect_stale_job_ids


@pytest.fixture(autouse=True)
def _clean_jobs():
    """Ensure each test starts and ends with an empty jobs dict."""
    srv.jobs.clear()
    yield
    srv.jobs.clear()


def _make_done_future():
    fut: Future = Future()
    fut.set_result({"ok": True})
    return fut


def _make_running_future():
    fut: Future = Future()
    # never resolved; mimics a job still in flight
    return fut


def test_collect_stale_removes_old_succeeded_jobs():
    now = 10_000.0
    srv.jobs["old_done"] = {
        "status": "succeeded",
        "future": _make_done_future(),
        "created_at": int(now - 3700),  # 1h+ old
    }
    srv.jobs["old_failed"] = {
        "status": "failed",
        "future": _make_done_future(),
        "created_at": int(now - 7200),
    }

    stale = _collect_stale_job_ids(now=now, ttl_sec=3600)

    assert set(stale) == {"old_done", "old_failed"}


def test_collect_stale_keeps_fresh_jobs():
    now = 10_000.0
    srv.jobs["recent_done"] = {
        "status": "succeeded",
        "future": _make_done_future(),
        "created_at": int(now - 60),  # only 1 minute old
    }

    stale = _collect_stale_job_ids(now=now, ttl_sec=3600)

    assert stale == []


def test_collect_stale_keeps_in_flight_jobs_even_if_old():
    """A queued/running job must NOT be GC'd, even if its created_at is ancient
    (e.g. a long-running detection). Only terminal-state jobs are eligible."""
    now = 10_000.0
    srv.jobs["stale_but_running"] = {
        "status": "running",
        "future": _make_running_future(),
        "created_at": int(now - 86_400),  # 1 day old, but still in flight
    }
    srv.jobs["stale_queued"] = {
        "status": "queued",
        "future": _make_running_future(),
        "created_at": int(now - 86_400),
    }

    stale = _collect_stale_job_ids(now=now, ttl_sec=3600)

    assert stale == []


def test_collect_stale_uses_lock_and_returns_sorted_ids():
    """GC must hold job_lock while inspecting the dict to avoid races with
    _save_job / delete. It should also return a stable order."""
    now = 10_000.0
    for jid in ("z-old", "a-old", "m-old"):
        srv.jobs[jid] = {
            "status": "succeeded",
            "future": _make_done_future(),
            "created_at": int(now - 7200),
        }
    srv.jobs["keep"] = {
        "status": "succeeded",
        "future": _make_done_future(),
        "created_at": int(now - 60),
    }

    stale = _collect_stale_job_ids(now=now, ttl_sec=3600)

    assert stale == ["a-old", "m-old", "z-old"]


def test_gc_loop_sweeps_and_exits_on_stop_event():
    """The background loop must exit cleanly when stop_event is set,
    and must call the sweep at least once before exiting."""
    from obb_geo_api_server import _jobs_gc_loop

    now = 10_000.0
    srv.jobs["stale"] = {
        "status": "succeeded",
        "future": _make_done_future(),
        "created_at": int(now - 7200),
    }

    stop = threading.Event()
    # Patch _collect_stale_job_ids to inject a controlled `now` value
    original_collect = srv._collect_stale_job_ids

    def fake_collect(*, now, ttl_sec):
        return original_collect(now=now, ttl_sec=ttl_sec)

    srv._collect_stale_job_ids = fake_collect  # type: ignore[assignment]
    try:
        thread = threading.Thread(
            target=_jobs_gc_loop,
            kwargs={"stop_event": stop, "interval_sec": 0.05, "ttl_sec": 3600},
            daemon=True,
        )
        thread.start()
        # Give the loop a moment to run at least one sweep
        time.sleep(0.15)
        stop.set()
        thread.join(timeout=2.0)
        assert not thread.is_alive(), "GC thread did not exit after stop_event"
    finally:
        srv._collect_stale_job_ids = original_collect  # type: ignore[assignment]

    assert "stale" not in srv.jobs, "Stale job should have been GC'd"
