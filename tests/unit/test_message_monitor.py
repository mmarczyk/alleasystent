"""Unit tests for services/message_monitor.py — the server-side pass that
notifies about unread buyer messages even when no browser tab is open."""
from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest


@pytest.fixture(autouse=True)
def set_env(monkeypatch):
    monkeypatch.setenv("GOOGLE_API_KEY", "test-key")
    monkeypatch.delenv("REDIS_URL", raising=False)


class FakeRedis:
    """Minimal stand-in for the redis.asyncio client surface this module uses."""

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


def _thread(tid: str, unread: bool = True, at: str = "2026-08-26T01:00:00Z"):
    """A thread shaped like Allegro's real /messaging/threads response.

    The field names matter more than they look: `read` and `lastMessageDateTime`
    are what Allegro actually sends. An earlier version of this file invented
    `hasUnreadMessages`/`lastMessageCreatedAt` to match the (equally wrong)
    production code, so the whole suite passed green while the monitor saw zero
    unread threads forever and never sent a single notification.
    """
    return {"id": tid, "read": not unread, "lastMessageDateTime": at}


class TestMarker:
    def test_marker_includes_last_message_timestamp(self):
        from services.message_monitor import _marker
        assert _marker(_thread("t1", at="2026-08-26T01:00:00Z")) == "t1@2026-08-26T01:00:00Z"

    def test_second_message_in_same_thread_is_a_different_marker(self):
        """A buyer writing twice must be announced twice — the browser-side
        monitor keyed on thread id alone and silently swallowed the follow-up."""
        from services.message_monitor import _marker
        assert _marker(_thread("t1", at="01:00")) != _marker(_thread("t1", at="02:00"))

    def test_missing_timestamp_does_not_crash(self):
        from services.message_monitor import _marker
        assert _marker({"id": "t1"}) == "t1@"


class TestDiffAndRecord:
    async def test_first_pass_records_baseline_without_reporting(self):
        from services.message_monitor import _diff_and_record
        r = FakeRedis()
        assert await _diff_and_record(r, "seen", ["t1@a", "t2@b"]) == []
        assert r.sets["seen"] == {"t1@a", "t2@b"}

    async def test_only_unseen_markers_are_new(self):
        from services.message_monitor import _diff_and_record
        r = FakeRedis({"seen": {"t1@a"}})
        assert await _diff_and_record(r, "seen", ["t1@a", "t2@b"]) == ["t2@b"]

    async def test_seen_set_is_replaced_not_accumulated(self):
        from services.message_monitor import _diff_and_record
        r = FakeRedis({"seen": {"old@x"}})
        await _diff_and_record(r, "seen", ["t1@a"])
        assert r.sets["seen"] == {"t1@a"}

    async def test_no_unread_leaves_state_untouched(self):
        """All threads read: nothing new, and the baseline must survive so a
        later message in a previously-seen thread still counts as new."""
        from services.message_monitor import _diff_and_record
        r = FakeRedis({"seen": {"t1@a"}})
        assert await _diff_and_record(r, "seen", []) == []
        assert r.sets["seen"] == {"t1@a"}

    async def test_ttl_is_set(self):
        from services.message_monitor import _diff_and_record, _SEEN_TTL
        r = FakeRedis()
        await _diff_and_record(r, "seen", ["t1@a"])
        assert r.expires["seen"] == _SEEN_TTL


class TestPollUser:
    async def _run(self, threads, seen=None, has_tokens=True):
        import services.message_monitor as mm

        r = FakeRedis(seen or {})
        r.exists = AsyncMock(side_effect=lambda key: 1 if key.startswith("allegro:tokens:") and has_tokens
                             else (1 if key in r.sets else 0))

        allegro = MagicMock()
        allegro._tokens = {"access_token": "x"} if has_tokens else None
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_message_threads = AsyncMock(return_value=threads)

        with patch("services.allegro_service.AllegroService") as svc, \
             patch.object(mm, "_notify", new=AsyncMock()) as notify:
            svc.get_instance.return_value = allegro
            await mm._poll_user(r, "user-1")
        return notify, allegro

    async def test_unread_thread_notifies(self):
        notify, _ = await self._run([_thread("t1")], seen={"allegro:monitor:messages:seen:user-1": {"t0@z"}})
        notify.assert_awaited_once()
        assert notify.await_args.kwargs["count"] == 1

    async def test_read_threads_are_ignored(self):
        notify, _ = await self._run(
            [_thread("t1", unread=False)],
            seen={"allegro:monitor:messages:seen:user-1": {"t0@z"}},
        )
        notify.assert_not_awaited()

    async def test_already_seen_thread_does_not_renotify(self):
        notify, _ = await self._run(
            [_thread("t1", at="2026-08-26T01:00:00Z")],
            seen={"allegro:monitor:messages:seen:user-1": {"t1@2026-08-26T01:00:00Z"}},
        )
        notify.assert_not_awaited()

    async def test_first_pass_does_not_notify_about_existing_unread(self):
        """Switching a user to server-side detection must not dump every
        already-unread thread on them as a push."""
        notify, _ = await self._run([_thread("t1"), _thread("t2")])
        notify.assert_not_awaited()

    async def test_user_without_allegro_tokens_is_skipped(self):
        notify, allegro = await self._run([_thread("t1")], has_tokens=False)
        allegro.get_message_threads.assert_not_awaited()
        notify.assert_not_awaited()

    async def test_allegro_api_error_is_swallowed(self):
        import services.message_monitor as mm
        from services.allegro_service import AllegroAPIError

        r = FakeRedis()
        r.exists = AsyncMock(return_value=1)
        allegro = MagicMock()
        allegro._tokens = {"access_token": "x"}
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_message_threads = AsyncMock(side_effect=AllegroAPIError(500, "boom"))

        with patch("services.allegro_service.AllegroService") as svc, \
             patch.object(mm, "_notify", new=AsyncMock()) as notify:
            svc.get_instance.return_value = allegro
            await mm._poll_user(r, "user-1")
        notify.assert_not_awaited()


class TestNotify:
    async def test_singular_and_plural_copy(self):
        import services.message_monitor as mm

        with patch("services.push_service.add_notification", new=AsyncMock(return_value=None)), \
             patch("services.push_service.send_push", new=AsyncMock()) as push:
            await mm._notify("user-1", count=1)
            assert push.await_args.kwargs["title"] == "Nowa wiadomość na Allegro"
            await mm._notify("user-1", count=3)
            assert push.await_args.kwargs["title"] == "3 nowych wiadomości na Allegro"

    async def test_push_carries_inbox_url_and_chat_prompt(self):
        import services.message_monitor as mm

        entry = {"id": "n1", "created_at": "2026-08-26T01:00:00Z"}
        with patch("services.push_service.add_notification", new=AsyncMock(return_value=entry)), \
             patch("services.push_service.send_push", new=AsyncMock()) as push:
            await mm._notify("user-1", count=1)
        kwargs = push.await_args.kwargs
        assert kwargs["url"] == "/?open=notifications"
        assert kwargs["prompt"] == "Pokaż mi tę nową wiadomość od kupującego."
        assert kwargs["notif_id"] == "n1"
        assert kwargs["created_at"] == entry["created_at"]


class TestEnabledFlag:
    async def test_reuses_the_existing_message_monitor_flag(self):
        """The flag the browser toggle has always written — switching detection
        to the server must not orphan users who already turned it on."""
        import services.message_monitor as mm

        with patch("services.monitor_state.is_monitor_enabled", new=AsyncMock(return_value=True)) as chk:
            assert await mm.is_monitor_enabled("user-1") is True
        assert chk.await_args.args == ("message", "user-1")

    async def test_run_once_without_redis_is_a_no_op(self):
        import services.message_monitor as mm
        with patch.object(mm, "_poll_all_users", new=AsyncMock()) as poll:
            await mm.run_once()
        poll.assert_not_awaited()


class TestScanKeyMatchesFlagKey:
    """The scan pattern in _poll_all_users must match the key monitor_state
    actually writes — a mismatch here is invisible and means every enabled
    user is silently skipped, which is the exact failure this module fixes."""

    def test_scan_pattern_matches_written_flag_key(self):
        import services.message_monitor as mm
        from services.monitor_state import _ENABLED_KEY
        from fnmatch import fnmatch

        written = _ENABLED_KEY.format(kind=mm._MONITOR_KIND, user_id="user-1")
        pattern = f"allegro:{mm._MONITOR_KIND}_monitor:enabled:*"
        assert fnmatch(written, pattern)
        assert written.count(":") >= 3 and written.split(":")[3] == "user-1"


class TestAllegroThreadSchema:
    """Pin the unread/timestamp field names against Allegro's real payload.

    This is the regression that mattered: /allegro/unread-messages and the
    monitor both read `hasUnreadMessages`/`lastMessageCreatedAt`, which Allegro
    never sends. `.get()` returns None instead of raising, so both saw zero
    unread threads forever and no notification was ever sent — with a green
    test suite, because the fixtures used the invented names too.
    """

    def test_unread_thread_from_real_payload_is_detected(self):
        from services.allegro_service import is_thread_unread
        assert is_thread_unread({"id": "t1", "read": False}) is True

    def test_read_thread_is_not_unread(self):
        from services.allegro_service import is_thread_unread
        assert is_thread_unread({"id": "t1", "read": True}) is False

    def test_thread_missing_the_field_counts_as_read(self):
        """Fail closed: a schema surprise must not flip every thread to "new"."""
        from services.allegro_service import is_thread_unread
        assert is_thread_unread({"id": "t1"}) is False

    def test_phantom_field_is_not_honoured(self):
        """The exact shape of the old bug, now asserted against."""
        from services.allegro_service import is_thread_unread
        assert is_thread_unread({"id": "t1", "hasUnreadMessages": True}) is False

    def test_last_message_timestamp_reads_allegro_field(self):
        from services.allegro_service import thread_last_message_at
        assert thread_last_message_at(
            {"id": "t1", "lastMessageDateTime": "2026-08-31T02:14:00Z"}
        ) == "2026-08-31T02:14:00Z"
        assert thread_last_message_at({"id": "t1", "lastMessageCreatedAt": "x"}) == ""

    async def test_realistic_thread_payload_reaches_notify(self):
        """End-to-end over _poll_user with a thread shaped like the real API."""
        import services.message_monitor as mm

        thread = {
            "id": "t-real",
            "read": False,
            "interlocutor": {"login": "kupujacy123"},
            "lastMessageDateTime": "2026-08-31T02:14:00Z",
        }
        r = FakeRedis({"allegro:monitor:messages:seen:user-1": {"t-old@2026-08-30T00:00:00Z"}})
        r.exists = AsyncMock(side_effect=lambda key: 1 if key.startswith("allegro:tokens:")
                             else (1 if key in r.sets else 0))
        allegro = MagicMock()
        allegro._tokens = {"access_token": "x"}
        allegro._load_tokens_from_redis = AsyncMock()
        allegro.get_message_threads = AsyncMock(return_value=[thread])

        with patch("services.allegro_service.AllegroService") as svc, \
             patch.object(mm, "_notify", new=AsyncMock()) as notify:
            svc.get_instance.return_value = allegro
            await mm._poll_user(r, "user-1")
        notify.assert_awaited_once()
        assert notify.await_args.kwargs["count"] == 1


class TestSingleSourceOfTruth:
    """The monitor, the HTTP endpoint and the chat agent must all decide
    "is this unread?" the same way — they disagreed before, and the two that
    were wrong failed silently."""

    def test_monitor_endpoint_and_agent_share_the_predicate(self):
        import inspect
        import services.message_monitor as mm
        import main
        from agents.allegro import allegro_agent

        for module in (mm, main, allegro_agent):
            src = inspect.getsource(module)
            assert "hasUnreadMessages" not in src, module.__name__
            assert "lastMessageCreatedAt" not in src, module.__name__
            assert "is_thread_unread" in src, module.__name__
