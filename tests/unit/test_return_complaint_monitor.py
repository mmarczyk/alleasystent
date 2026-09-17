"""Unit tests for services/return_complaint_monitor.py.

Focused on the baseline rule, which is where the message monitor lost a real
notification: the key that marks "we have already seen this user's inbox" was
only written on a pass that found something, so the first return or complaint
a user ever got was recorded as the baseline instead of being announced.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


class FakeRedis:
    """Minimal stand-in for the redis.asyncio surface this module uses."""

    def __init__(self, sets: dict | None = None, keys: dict | None = None):
        self.sets = {k: set(v) for k, v in (sets or {}).items()}
        self.keys = dict(keys or {})
        self.expires: dict[str, int] = {}

    async def exists(self, key):
        return 1 if key in self.sets or key in self.keys else 0

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

    async def set(self, key, value, ex=None, nx=False):
        """Mirrors redis-py: SET NX returns None when the key already exists."""
        if nx and key in self.keys:
            return None
        self.keys[key] = value
        if ex is not None:
            self.expires[key] = ex
        return True

    async def delete(self, key):
        self.sets.pop(key, None)
        self.keys.pop(key, None)

    def pipeline(self):
        return FakePipeline(self)


class FakePipeline:
    def __init__(self, redis: FakeRedis):
        self.redis = redis
        self.ops: list = []

    def delete(self, key):
        self.ops.append(("delete", key))

    def sadd(self, key, *members):
        self.ops.append(("sadd", key, members))

    def expire(self, key, ttl):
        self.ops.append(("expire", key, ttl))

    async def execute(self):
        for op in self.ops:
            if op[0] == "delete":
                self.redis.sets.pop(op[1], None)
            elif op[0] == "sadd":
                self.redis.sets.setdefault(op[1], set()).update(op[2])
            elif op[0] == "expire":
                self.redis.expires[op[1]] = op[2]
        self.ops = []


class TestDiffAndRecord:
    async def test_first_pass_records_baseline_without_reporting(self):
        from services.return_complaint_monitor import _diff_and_record
        r = FakeRedis()
        assert await _diff_and_record(r, "seen", ["r1", "r2"]) == []
        assert await r.exists("seen") == 1

    async def test_only_unseen_ids_are_new(self):
        from services.return_complaint_monitor import _diff_and_record
        r = FakeRedis({"seen": {"r1"}})
        assert await _diff_and_record(r, "seen", ["r1", "r2"]) == ["r2"]

    async def test_empty_first_pass_still_creates_the_key(self):
        """So the user's first-ever return is announced, not swallowed."""
        from services.return_complaint_monitor import _diff_and_record
        r = FakeRedis()
        assert await _diff_and_record(r, "seen", []) == []
        assert await r.exists("seen") == 1

    async def test_first_return_after_quiet_passes_is_reported(self):
        from services.return_complaint_monitor import _diff_and_record
        r = FakeRedis()
        assert await _diff_and_record(r, "seen", []) == []       # nothing yet
        assert await _diff_and_record(r, "seen", ["r1"]) == ["r1"]

    async def test_empty_pass_keeps_recorded_ids_and_refreshes_ttl(self):
        from services.return_complaint_monitor import _diff_and_record, _SEEN_TTL
        r = FakeRedis({"seen": {"r1"}})
        assert await _diff_and_record(r, "seen", []) == []
        assert "r1" in r.sets["seen"]
        assert r.expires["seen"] == _SEEN_TTL

    async def test_seen_set_is_replaced_not_accumulated(self):
        from services.return_complaint_monitor import _diff_and_record, _BASELINE_MEMBER
        r = FakeRedis({"seen": {"old"}})
        await _diff_and_record(r, "seen", ["r1"])
        assert r.sets["seen"] == {_BASELINE_MEMBER, "r1"}

    async def test_baseline_member_never_reported_as_new(self):
        """The sentinel lives in the set; it must never look like a real ID."""
        from services.return_complaint_monitor import _diff_and_record, _BASELINE_MEMBER
        r = FakeRedis({"seen": {_BASELINE_MEMBER}})
        assert await _diff_and_record(r, "seen", ["r1"]) == ["r1"]


class TestToProcessBaseline:
    """The to-process list is a TODO, not a feed: what is already waiting is
    exactly what the seller was never told about."""

    async def test_first_pass_reports_instead_of_baselining(self):
        from services.return_complaint_monitor import _diff_and_record
        r = FakeRedis()
        new = await _diff_and_record(r, "todo", ["r1", "r2"], baseline_first_pass=False)
        assert new == ["r1", "r2"]

    async def test_baselining_is_still_the_default(self):
        from services.return_complaint_monitor import _diff_and_record
        r = FakeRedis()
        assert await _diff_and_record(r, "seen", ["r1"]) == []

    async def test_second_pass_does_not_repeat_the_same_ids(self):
        from services.return_complaint_monitor import _diff_and_record
        r = FakeRedis()
        await _diff_and_record(r, "todo", ["r1"], baseline_first_pass=False)
        assert await _diff_and_record(r, "todo", ["r1"], baseline_first_pass=False) == []


class FakeAllegro:
    def __init__(self, to_process: list[dict]):
        self.to_process = to_process
        self.calls: list[dict] = []

    async def get_customer_returns(self, **kwargs):
        self.calls.append(kwargs)
        return self.to_process


class TestPollReturnsToProcess:
    """The pass that answers "mam zwrot nieobsłużony, a nie dostałem
    powiadomienia": a return awaiting a decision is announced even though
    nothing about it is new, and re-announced daily while it waits."""

    async def test_waiting_return_is_announced_on_the_very_first_pass(self):
        from unittest.mock import AsyncMock, patch
        import services.return_complaint_monitor as m

        r = FakeRedis()
        allegro = FakeAllegro([{"id": "r1"}])
        with patch.object(m, "_notify", AsyncMock()) as notify:
            await m._poll_returns_to_process(r, allegro, "u1")
        notify.assert_awaited_once_with("u1", kind="returns_to_process", count=1)

    async def test_it_asks_allegro_only_for_returns_awaiting_a_decision(self):
        from unittest.mock import AsyncMock, patch
        import services.return_complaint_monitor as m

        allegro = FakeAllegro([])
        with patch.object(m, "_notify", AsyncMock()):
            await m._poll_returns_to_process(FakeRedis(), allegro, "u1")
        assert allegro.calls[0]["status"] == m._TO_PROCESS_STATUS

    async def test_same_return_is_not_announced_again_within_the_window(self):
        from unittest.mock import AsyncMock, patch
        import services.return_complaint_monitor as m

        r = FakeRedis()
        allegro = FakeAllegro([{"id": "r1"}])
        with patch.object(m, "_notify", AsyncMock()) as notify:
            await m._poll_returns_to_process(r, allegro, "u1")
            await m._poll_returns_to_process(r, allegro, "u1")
            await m._poll_returns_to_process(r, allegro, "u1")
        assert notify.await_count == 1

    async def test_still_unhandled_return_is_re_announced_once_the_window_expires(self):
        from unittest.mock import AsyncMock, patch
        import services.return_complaint_monitor as m

        r = FakeRedis()
        allegro = FakeAllegro([{"id": "r1"}])
        with patch.object(m, "_notify", AsyncMock()) as notify:
            await m._poll_returns_to_process(r, allegro, "u1")
            # The reminder key expiring is the whole cadence — simulate the day passing.
            r.keys.pop(m._TO_PROCESS_REMINDED_KEY.format(user_id="u1"))
            await m._poll_returns_to_process(r, allegro, "u1")
        assert notify.await_count == 2
        assert notify.await_args.kwargs["repeat"] is True

    async def test_a_second_waiting_return_is_announced_immediately(self):
        from unittest.mock import AsyncMock, patch
        import services.return_complaint_monitor as m

        r = FakeRedis()
        with patch.object(m, "_notify", AsyncMock()) as notify:
            await m._poll_returns_to_process(r, FakeAllegro([{"id": "r1"}]), "u1")
            await m._poll_returns_to_process(r, FakeAllegro([{"id": "r1"}, {"id": "r2"}]), "u1")
        assert notify.await_count == 2
        assert notify.await_args.kwargs["count"] == 1  # only the new one

    async def test_handling_everything_clears_the_reminder_window(self):
        from unittest.mock import AsyncMock, patch
        import services.return_complaint_monitor as m

        r = FakeRedis()
        remind_key = m._TO_PROCESS_REMINDED_KEY.format(user_id="u1")
        with patch.object(m, "_notify", AsyncMock()) as notify:
            await m._poll_returns_to_process(r, FakeAllegro([{"id": "r1"}]), "u1")
            await m._poll_returns_to_process(r, FakeAllegro([]), "u1")
            assert remind_key not in r.keys
            # …so the next one to arrive is announced at once, not a day later.
            await m._poll_returns_to_process(r, FakeAllegro([{"id": "r2"}]), "u1")
        assert notify.await_count == 2

    async def test_api_error_notifies_nothing_and_records_nothing(self):
        from unittest.mock import AsyncMock, patch
        import services.return_complaint_monitor as m
        from services.allegro_service import AllegroAPIError

        class Failing:
            async def get_customer_returns(self, **kwargs):
                raise AllegroAPIError(500, "boom")

        r = FakeRedis()
        with patch.object(m, "_notify", AsyncMock()) as notify:
            await m._poll_returns_to_process(r, Failing(), "u1")
        notify.assert_not_awaited()
        assert r.sets == {}


class TestNotifyWording:
    async def test_polish_plural_forms(self):
        from services.return_complaint_monitor import _plural_pl
        assert _plural_pl(1, "zwrot", "zwroty", "zwrotów") == "zwrot"
        assert _plural_pl(3, "zwrot", "zwroty", "zwrotów") == "zwroty"
        assert _plural_pl(5, "zwrot", "zwroty", "zwrotów") == "zwrotów"
        assert _plural_pl(12, "zwrot", "zwroty", "zwrotów") == "zwrotów"
        assert _plural_pl(22, "zwrot", "zwroty", "zwrotów") == "zwroty"

    async def test_repeat_says_the_return_is_still_waiting(self):
        from unittest.mock import AsyncMock, patch
        import services.return_complaint_monitor as m

        with patch("services.push_service.add_notification", AsyncMock(return_value=None)), \
             patch("services.push_service.send_push", AsyncMock()) as push:
            await m._notify("u1", kind="returns_to_process", count=1, repeat=True)
        title = push.await_args.kwargs["title"]
        assert "wciąż" in title
        assert push.await_args.kwargs["prompt"] == "Pokaż mi zwroty do obsłużenia."

