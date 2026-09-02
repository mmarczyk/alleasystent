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
    """Minimal stand-in for the redis.asyncio surface _diff_and_record uses."""

    def __init__(self, sets: dict | None = None):
        self.sets = {k: set(v) for k, v in (sets or {}).items()}
        self.expires: dict[str, int] = {}

    async def exists(self, key):
        return 1 if key in self.sets else 0

    async def smembers(self, key):
        return set(self.sets.get(key, set()))

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
