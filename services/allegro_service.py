from __future__ import annotations

"""Allegro REST API client with OAuth2 device-flow authentication."""

import asyncio
import base64
import json
import logging
from datetime import datetime, timedelta, timezone
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from config.settings import get_settings
from models.allegro import AllegroInvoiceBuyer, AllegroOrder, AllegroOrderLine, AllegroTokens

logger = logging.getLogger(__name__)

# How many per-order invoice lookups (GET .../invoices) may be in flight at once.
# Allegro rate-limits per second, and the seller is waiting on the answer — 8 is
# fast enough for a few hundred orders without hammering the API.
_INVOICE_LOOKUP_CONCURRENCY = 8

# Statuses worth a second attempt: the request was fine, Allegro just couldn't
# serve it now. 0 is our own marker for a network error/timeout (see _get).
_RETRYABLE_STATUSES = frozenset({0, 429, 500, 502, 503, 504})

# Billing listings are fetched one month-sized window at a time rather than as a
# single year-long range: a yearly sales summary is thousands of entries, and
# one long chain of pages is one 429 or timeout away from losing the whole cost
# report. Windows are independent, so a hiccup costs a retry of one window.
_BILLING_WINDOW_DAYS = 31
# Per window: 200 pages × 100 entries. A month of billing never comes close —
# the cap only exists so a paging bug can't spin forever.
_BILLING_MAX_PAGES = 200

# Allegro grants a token exactly the scopes the app was authorized with, and
# rejects anything outside them with 403 — so a missing scope looks like a
# permission problem on the account, which it isn't. Naming the scope a call
# needs lets the 403 say which one is absent instead of leaving it to guesswork.
SCOPE_BILLING_READ = "allegro:api:billing:read"
SCOPE_ORDERS_WRITE = "allegro:api:orders:write"

# Allegro's own limit on an order's invoice attachment.
INVOICE_FILE_MAX_BYTES = 3 * 1024 * 1024


def decode_token_scopes(access_token: str | None) -> list[str] | None:
    """Scopes carried by an Allegro access token, or None if they can't be read.

    Allegro issues JWTs, and the payload's `scope` claim is the authoritative
    answer to "may this token do that" — asking it locally turns a bare 403
    into a message that names the missing permission. The signature is
    irrelevant here (Allegro already validated it; this only reports what the
    token says about itself), so it is not checked. Anything unparseable —
    a mock token, a format change — reads as None, i.e. "unknown", never as
    "no scopes".
    """
    if not access_token:
        return None
    parts = access_token.split(".")
    if len(parts) != 3:
        return None
    payload_b64 = parts[1]
    try:
        padded = payload_b64 + "=" * (-len(payload_b64) % 4)
        payload = json.loads(base64.urlsafe_b64decode(padded))
    except Exception:
        return None
    scope = payload.get("scope")
    if isinstance(scope, str):
        return scope.split()
    if isinstance(scope, list):
        return [str(s) for s in scope]
    return None


def parse_invoice_buyer(checkout_form: dict[str, Any]) -> AllegroInvoiceBuyer:
    """Read the VAT-invoice address block off a checkout form.

    Shared by _parse_order (list responses) and get_order_invoice_data (single
    order) so both read the same fields the same way — the company/private-person
    distinction the invoicing flow and get_buyers both rest on must not depend on
    which endpoint the order happened to arrive from.

    Allegro's tax ID field has moved twice: legacy "taxId" is deprecated in favor
    of an "ids" array (e.g. [{"type": "PL_NIP", "value": "..."}]) — "vatId" was
    never a real field on either shape. Prefer the current PL_NIP entry, fall
    back to the deprecated field for older responses.
    """
    invoice = checkout_form.get("invoice") or {}
    address = invoice.get("address") or {}
    company = address.get("company") or {}
    natural = address.get("naturalPerson") or {}
    tax_ids = company.get("ids") or []
    vat_id = next((i.get("value", "") for i in tax_ids if i.get("type") == "PL_NIP"), "")
    vat_id = vat_id or company.get("taxId", "")
    return AllegroInvoiceBuyer(
        required=bool(invoice.get("required")),
        dont_want=bool(invoice.get("dontWant")),
        company_name=company.get("name", "") or "",
        vat_id=vat_id or "",
        first_name=natural.get("firstName", "") or "",
        last_name=natural.get("lastName", "") or "",
        street=address.get("street", "") or "",
        city=address.get("city", "") or "",
        zip_code=address.get("zipCode", "") or "",
        country_code=address.get("countryCode", "") or "",
    )


class _TTLCache:
    """Minimal in-memory TTL cache. Thread-safe enough for single-process async use."""

    def __init__(self, ttl: float):
        self._ttl = ttl
        self._store: dict[str, tuple[Any, float]] = {}

    def get(self, key: str) -> Any:
        entry = self._store.get(key)
        if entry is None:
            return None
        value, ts = entry
        if monotonic() - ts > self._ttl:
            del self._store[key]
            return None
        return value

    def set(self, key: str, value: Any) -> None:
        self._store[key] = (value, monotonic())

    def invalidate(self, key: str) -> None:
        self._store.pop(key, None)

    def clear(self) -> None:
        self._store.clear()


def _parse_iso_utc(value: str | None) -> datetime | None:
    """Parse an Allegro ISO 8601 timestamp into an aware datetime, or None."""
    if not value:
        return None
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _in_utc_period(value: str | None, date_from: str, date_to: str) -> bool:
    """True when `value` falls inside [date_from, date_to] (UTC ISO 8601 bounds).

    Compared as instants, not strings: an item timestamped '+02:00' and a bound
    written as 'Z' are the same clock read two ways, and a lexicographic
    comparison of the two would silently drop it from the period.
    """
    moment = _parse_iso_utc(value)
    if moment is None:
        return False
    start, end = _parse_iso_utc(date_from), _parse_iso_utc(date_to)
    if start is None or end is None:
        return False
    return start <= moment <= end


def return_created_at(item: dict[str, Any]) -> str | None:
    """When a customer return was reported. Read defensively across the
    candidate keys — the beta resource's exact shape isn't fully documented
    (AllegroAgent._return_bullet reads the same set for display)."""
    reception = item.get("reception") if isinstance(item.get("reception"), dict) else {}
    return item.get("createdAt") or item.get("receivedAt") or reception.get("createdAt")


def issue_opened_at(item: dict[str, Any]) -> str | None:
    """When a dispute/claim was opened — openedDate per Allegro's /sale/issues docs."""
    return item.get("openedDate") or item.get("createdAt")


class AllegroAuthError(Exception):
    pass


async def exchange_allegro_code(code: str, redirect_uri: str | None = None) -> tuple[str, "AllegroTokens"]:
    """
    Exchange Allegro authorization code for tokens.
    Returns (allegro_login, tokens) — login is used as the user_id.
    """
    settings = get_settings()
    async with httpx.AsyncClient(timeout=30.0) as client:
        resp = await client.post(
            f"{settings.allegro_auth_url}/token",
            auth=(settings.allegro_client_id, settings.allegro_client_secret),
            data={
                "grant_type": "authorization_code",
                "code": code,
                "redirect_uri": redirect_uri or settings.allegro_redirect_uri,
            },
        )
        resp.raise_for_status()
        data = resp.json()
        tokens = AllegroTokens(
            access_token=data["access_token"],
            refresh_token=data["refresh_token"],
            expires_at=datetime.utcnow() + timedelta(seconds=data["expires_in"] - 60),
            token_type=data.get("token_type", "Bearer"),
        )
        me = await client.get(
            f"{settings.allegro_api_url}/me",
            headers={
                "Authorization": f"Bearer {tokens.access_token}",
                "Accept": "application/vnd.allegro.public.v1+json",
            },
        )
        me.raise_for_status()
        login: str = me.json().get("login", "unknown")
    return login, tokens


class AllegroAPIError(Exception):
    def __init__(self, status_code: int, detail: str, code: str = "", user_message: str = ""):
        self.status_code = status_code
        self.code = code
        self.user_message = user_message
        super().__init__(f"Allegro API error {status_code}: {detail}")


def _api_error(response: httpx.Response) -> AllegroAPIError:
    """Turn an error response into an AllegroAPIError carrying Allegro's own wording.

    Allegro answers failures with {"errors": [{"code", "message", "userMessage"}]}.
    Dumping that JSON verbatim at the seller ("Allegro API error 403: {...}")
    hides the one sentence that explains the problem, so the parsed userMessage
    becomes the exception's detail and stays available on the exception for
    callers that build their own message. Anything unparseable falls back to
    the raw body, which is what this always used to show.
    """
    raw = response.text[:500]
    try:
        errors = response.json().get("errors") or []
        first = errors[0] if isinstance(errors, list) and errors else {}
    except Exception:
        first = {}
    if not isinstance(first, dict):
        first = {}
    code = str(first.get("code") or "")
    user_message = str(first.get("userMessage") or first.get("message") or "")
    detail = user_message or raw
    if code and user_message:
        detail = f"{user_message} ({code})"
    return AllegroAPIError(response.status_code, detail, code=code, user_message=user_message)


def is_thread_unread(thread: dict[str, Any]) -> bool:
    """Whether an Allegro messaging thread has unread buyer messages.

    Allegro's /messaging/threads marks this with a `read` boolean — there is no
    `hasUnreadMessages` field. Reading it wrong is silent: the caller just sees
    zero unread forever instead of erroring, which is exactly how
    /allegro/unread-messages and the message monitor both went months without
    ever firing a notification. Hence one shared predicate next to the API call
    rather than the field name spelled out at each call site.

    A thread missing the field counts as read, so a schema change can't flip
    every thread to "new" and spam the user.
    """
    return not thread.get("read", True)


def thread_last_message_at(thread: dict[str, Any]) -> str:
    """Timestamp of a thread's most recent message (Allegro: lastMessageDateTime)."""
    return thread.get("lastMessageDateTime") or ""


class AllegroService:
    """
    Wraps the Allegro REST API.

    Authentication uses OAuth2 device flow (suitable for server-side apps).
    Tokens are refreshed automatically before expiry.
    Token persistence: Redis when REDIS_URL is set (survives redeployments),
    otherwise local file fallback.

    One instance is kept per user_id for the lifetime of the process so that
    the httpx connection pool, in-memory caches, and loaded tokens are reused
    across requests instead of being rebuilt from scratch every time.
    """

    _instances: dict[str, "AllegroService"] = {}

    @classmethod
    def get_instance(cls, user_id: str | None = None) -> "AllegroService":
        key = user_id or "default"
        if key not in cls._instances:
            cls._instances[key] = cls(user_id)
        return cls._instances[key]

    def __init__(self, user_id: str | None = None):
        self._user_id = user_id or "default"
        self._settings = get_settings()
        self._tokens: AllegroTokens | None = None
        self._pending_device_code: str | None = None
        self._client = httpx.AsyncClient(
            base_url=self._settings.allegro_api_url,
            timeout=30.0,
        )
        self._redis = None
        self._init_redis()
        self._load_tokens()
        self._load_pending_device_code()
        # Mock mode: always override with a fake long-lived token so the agent
        # skips OAuth even when an expired real token exists on disk.
        if self._settings.allegro_mock_token:
            from datetime import datetime, timedelta
            self._tokens = AllegroTokens(
                access_token=self._settings.allegro_mock_token,
                refresh_token="mock-refresh",
                expires_at=datetime.utcnow() + timedelta(days=365),
            )
        # Single order details (buyer, items, price) — static once placed, 5 min TTL
        self._order_cache: _TTLCache = _TTLCache(ttl=300.0)
        # Invoice status per order — 2 min TTL
        self._invoice_cache: _TTLCache = _TTLCache(ttl=120.0)
        # Full offer catalogue — 5 min TTL (stock/prices change infrequently)
        self._all_offers_cache: _TTLCache = _TTLCache(ttl=300.0)

    @property
    def _device_code_file(self) -> str:
        return f".allegro_device_code_{self._user_id}"

    @property
    def _redis_tokens_key(self) -> str:
        return f"allegro:tokens:{self._user_id}"

    def _token_file(self) -> Path:
        if self._user_id == "default":
            return Path(self._settings.allegro_token_file)
        return Path(f".allegro_tokens_{self._user_id}.json")

    def _init_redis(self) -> None:
        if not self._settings.redis_url:
            return
        try:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._settings.redis_url, decode_responses=True)
            logger.info("AllegroService: Redis ready for token persistence")
        except Exception as exc:
            logger.warning("AllegroService: Redis unavailable (%s) — file-only token storage", exc)

    # ── Token management ──────────────────────────────────────────────────────

    def _load_tokens(self) -> None:
        path = self._token_file()
        if path.exists():
            try:
                data = json.loads(path.read_text())
                data["expires_at"] = datetime.fromisoformat(data["expires_at"])
                self._tokens = AllegroTokens(**data)
                logger.info("Loaded Allegro tokens from %s", path)
            except Exception as exc:
                logger.warning("Failed to load Allegro tokens: %s", exc)

    async def _load_tokens_from_redis(self) -> None:
        if self._redis is None:
            return
        try:
            raw = await self._redis.get(self._redis_tokens_key)
            if raw:
                data = json.loads(raw)
                data["expires_at"] = datetime.fromisoformat(data["expires_at"])
                self._tokens = AllegroTokens(**data)
                logger.info("Loaded Allegro tokens from Redis")
        except Exception as exc:
            logger.warning("Failed to load Allegro tokens from Redis: %s", exc)

    async def _save_tokens(self) -> None:
        if self._tokens is None:
            return
        data = self._tokens.model_dump()
        data["expires_at"] = data["expires_at"].isoformat()
        raw = json.dumps(data, indent=2)

        async def _write_file() -> None:
            try:
                await asyncio.to_thread(self._token_file().write_text, raw)
            except Exception as exc:
                logger.warning("Could not write token file: %s", exc)

        async def _write_redis() -> None:
            if self._redis is None:
                return
            try:
                # 90-day TTL so stale/revoked tokens don't linger forever
                await self._redis.set(self._redis_tokens_key, raw, ex=86400 * 90)
                logger.info("Saved Allegro tokens to Redis")
            except Exception as exc:
                logger.warning("Failed to save Allegro tokens to Redis: %s", exc)

        await asyncio.gather(_write_file(), _write_redis())

    def _load_pending_device_code(self) -> None:
        path = Path(self._device_code_file)
        if path.exists():
            try:
                self._pending_device_code = path.read_text().strip() or None
            except Exception as exc:
                logger.warning("Failed to load pending device code: %s", exc)

    def _save_pending_device_code(self) -> None:
        Path(self._device_code_file).write_text(self._pending_device_code or "")

    def _clear_pending_device_code(self) -> None:
        self._pending_device_code = None
        path = Path(self._device_code_file)
        if path.exists():
            path.unlink()

    async def start_device_flow(self) -> dict[str, str]:
        """
        Initiate device authorization flow.
        Returns dict with 'user_code', 'verification_uri', 'device_code'.
        The caller must display the code to the store owner who authorizes in browser.
        """
        async with httpx.AsyncClient(timeout=30.0) as client:
            resp = await client.post(
                f"{self._settings.allegro_auth_url}/device",
                auth=(self._settings.allegro_client_id, self._settings.allegro_client_secret),
                data={
                    "client_id": self._settings.allegro_client_id,
                    # billing:read is what /billing/billing-entries needs — without it
                    # every cost figure in the sales summary comes back 403.
                    "scope": "allegro:api:sale:offers:read allegro:api:orders:read "
                             "allegro:api:orders:write allegro:api:messaging "
                             "allegro:api:billing:read",
                },
            )
            # Allegro may return 302 with JSON body, or 200 directly
            if resp.status_code in (200, 302):
                try:
                    data = resp.json()
                    self._pending_device_code = data.get("device_code")
                    self._save_pending_device_code()
                    return data
                except Exception:
                    pass
            resp.raise_for_status()
            data = resp.json()
            self._pending_device_code = data.get("device_code")
            self._save_pending_device_code()
            return data

    async def poll_device_flow(self, device_code: str, interval: int = 5) -> bool:
        """Poll for token after user authorizes. Returns True when tokens obtained."""
        import asyncio

        deadline = datetime.utcnow() + timedelta(minutes=10)
        async with httpx.AsyncClient() as client:
            while datetime.utcnow() < deadline:
                await asyncio.sleep(interval)
                resp = await client.post(
                    f"{self._settings.allegro_auth_url}/token",
                    auth=(self._settings.allegro_client_id, self._settings.allegro_client_secret),
                    data={
                        "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                        "device_code": device_code,
                    },
                )
                if resp.status_code == 200:
                    data = resp.json()
                    self._tokens = AllegroTokens(
                        access_token=data["access_token"],
                        refresh_token=data["refresh_token"],
                        expires_at=datetime.utcnow() + timedelta(seconds=data["expires_in"] - 60),
                        token_type=data.get("token_type", "Bearer"),
                    )
                    await self._save_tokens()
                    self._clear_pending_device_code()
                    logger.info("Allegro tokens obtained via device flow")
                    return True
                error = resp.json().get("error", "")
                if error == "authorization_pending":
                    continue
                raise AllegroAuthError(f"Device flow error: {error}")
        return False

    async def try_complete_device_flow(self) -> bool:
        """
        Single poll attempt against the token endpoint.
        Returns True if tokens were obtained, False if still pending or device code missing.
        Raises AllegroAuthError on hard failures (expired, denied, etc.).
        """
        if not self._pending_device_code:
            return False
        async with httpx.AsyncClient(timeout=15.0) as client:
            resp = await client.post(
                f"{self._settings.allegro_auth_url}/token",
                auth=(self._settings.allegro_client_id, self._settings.allegro_client_secret),
                data={
                    "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                    "device_code": self._pending_device_code,
                },
            )
        if resp.status_code == 200:
            data = resp.json()
            self._tokens = AllegroTokens(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=datetime.utcnow() + timedelta(seconds=data["expires_in"] - 60),
                token_type=data.get("token_type", "Bearer"),
            )
            await self._save_tokens()
            self._clear_pending_device_code()
            logger.info("Allegro tokens obtained via device flow completion")
            return True
        error = resp.json().get("error", "")
        if error == "authorization_pending":
            return False
        raise AllegroAuthError(f"Device flow failed: {error}")

    async def _refresh_tokens(self) -> None:
        if self._tokens is None:
            raise AllegroAuthError("No tokens available. Run device flow first.")
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._settings.allegro_auth_url}/token",
                auth=(self._settings.allegro_client_id, self._settings.allegro_client_secret),
                data={
                    "grant_type": "refresh_token",
                    "refresh_token": self._tokens.refresh_token,
                },
            )
            if resp.status_code != 200:
                raise AllegroAuthError(f"Token refresh failed: {resp.text}")
            data = resp.json()
            self._tokens = AllegroTokens(
                access_token=data["access_token"],
                refresh_token=data["refresh_token"],
                expires_at=datetime.utcnow() + timedelta(seconds=data["expires_in"] - 60),
            )
            await self._save_tokens()
            logger.info("Allegro tokens refreshed")

    def token_scopes(self) -> list[str] | None:
        """Scopes on the currently loaded access token, or None when unknown."""
        if self._tokens is None:
            return None
        return decode_token_scopes(self._tokens.access_token)

    def has_scope(self, scope: str) -> bool | None:
        """True/False when the token's scopes can be read, None when they can't.

        Three-valued on purpose: "we cannot tell" must not be reported to the
        seller as "you are missing the permission".
        """
        scopes = self.token_scopes()
        if scopes is None:
            return None
        return scope in scopes

    async def _get_headers(self) -> dict[str, str]:
        if self._tokens is None:
            await self._load_tokens_from_redis()
        if self._tokens is None:
            raise AllegroAuthError("Not authenticated. Run device flow first.")
        if self._tokens.is_expired():
            await self._refresh_tokens()
        return {
            "Authorization": f"Bearer {self._tokens.access_token}",
            "Accept": "application/vnd.allegro.public.v1+json",
            "Content-Type": "application/vnd.allegro.public.v1+json",
        }

    async def _get(self, path: str, params: dict | list | None = None, accept: str | None = None) -> dict[str, Any]:
        headers = await self._get_headers()
        if accept:
            headers["Accept"] = accept
        try:
            resp = await self._client.get(path, headers=headers, params=params)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AllegroAPIError(0, f"Network error: {exc}") from exc
        if resp.status_code >= 400:
            raise _api_error(resp)
        return resp.json()

    async def _post(self, path: str, body: dict) -> dict[str, Any]:
        headers = await self._get_headers()
        try:
            resp = await self._client.post(path, headers=headers, json=body)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AllegroAPIError(0, f"Network error: {exc}") from exc
        if resp.status_code >= 400:
            raise _api_error(resp)
        return resp.json()

    async def _put_bytes(self, path: str, content: bytes, content_type: str) -> None:
        headers = await self._get_headers()
        headers["Content-Type"] = content_type
        try:
            resp = await self._client.put(path, headers=headers, content=content)
        except (httpx.ConnectError, httpx.TimeoutException) as exc:
            raise AllegroAPIError(0, f"Network error: {exc}") from exc
        if resp.status_code >= 400:
            raise _api_error(resp)

    # ── Orders ────────────────────────────────────────────────────────────────

    async def get_orders(
        self,
        status: str | None = None,
        buyer_login: str | None = None,
        fulfillment_status: str | None = None,
        line_items_sent: list[str] | None = None,
        bought_at_gte: str | None = None,
        bought_at_lte: str | None = None,
        paid_at_gte: str | None = None,
        paid_at_lte: str | None = None,
        limit: int = 100,
        offset: int = 0,
    ) -> list[AllegroOrder]:
        # Order lists are never cached — new orders can arrive at any time and
        # stale counts/statuses would be misleading.  Only individual order
        # details (get_order) are cached because they don't change once placed.

        # When filtering by payment time the Allegro API has no direct parameter —
        # fetch READY_FOR_PROCESSING orders with a broad boughtAt window (last 7 days)
        # and filter client-side by paid_at.
        if paid_at_gte or paid_at_lte:
            from datetime import datetime, timedelta
            from zoneinfo import ZoneInfo
            start_window = (
                datetime.now(ZoneInfo("UTC")) - timedelta(days=7)
            ).strftime("%Y-%m-%dT%H:%M:%SZ")
            raw = await self.get_orders(
                status="READY_FOR_PROCESSING",
                bought_at_gte=start_window,
                limit=200,
            )
            result = raw
            if paid_at_gte:
                result = [o for o in result if (o.paid_at or "") >= paid_at_gte]
            if paid_at_lte:
                result = [o for o in result if (o.paid_at or "") <= paid_at_lte]
            return result[:limit]

        base_params: dict[str, Any] = {}
        if status:
            base_params["status"] = status
        if buyer_login:
            base_params["buyer.login"] = buyer_login
        if fulfillment_status:
            base_params["fulfillment.status"] = fulfillment_status
        if line_items_sent:
            base_params["fulfillment.shipmentSummary.lineItemsSent"] = line_items_sent
        if bought_at_gte:
            base_params["lineItems.boughtAt.gte"] = bought_at_gte
        if bought_at_lte:
            base_params["lineItems.boughtAt.lte"] = bought_at_lte

        # Auto-paginate — Allegro returns up to 100 per page.
        # Scan a full page even when `limit` is small (e.g. limit=1 for "last order"):
        # the API's own ordering isn't guaranteed newest-first, so we need enough data
        # to sort ourselves below before truncating to `limit`.
        all_orders: list[AllegroOrder] = []
        page_size = 100
        scan_cap = max(limit, page_size)
        cur_offset = offset
        while len(all_orders) < scan_cap:
            params = {**base_params, "limit": page_size, "offset": cur_offset}
            data = await self._get("/order/checkout-forms", params=params)
            forms = data.get("checkoutForms", [])
            total_count = int(data.get("totalCount", 0))
            all_orders.extend(self._parse_order(o) for o in forms)
            if len(all_orders) >= total_count or len(forms) < page_size:
                break
            cur_offset += page_size

        # Newest-first, so a small `limit` reliably returns the most recent orders.
        all_orders.sort(key=lambda o: o.created_at or "", reverse=True)
        result = all_orders[:limit]
        # Populate the single-order detail cache as a side-effect so that
        # a subsequent get_order(id) for any of these doesn't need a round-trip.
        for order in result:
            self._order_cache.set(order.order_id, order)
        return result

    async def get_order(self, order_id: str) -> AllegroOrder:
        cached = self._order_cache.get(order_id)
        if cached is not None:
            logger.debug("order cache hit: %s", order_id)
            return cached
        data = await self._get(f"/order/checkout-forms/{order_id}")
        order = self._parse_order(data)
        self._order_cache.set(order_id, order)
        return order

    async def get_all_paid_orders_in_period(
        self,
        date_from: str,
        date_to: str,
    ) -> list[AllegroOrder]:
        """Fetch paid orders where payment.finishedAt falls in [date_from, date_to] (both UTC ISO strings).

        Allegro API only supports boughtAt filtering, so we fetch a wider window
        and filter client-side by payment.finishedAt.
        """
        all_fetched: list[AllegroOrder] = []
        page_size = 50
        offset = 0
        while True:
            params: dict[str, Any] = {
                "status": "READY_FOR_PROCESSING",
                "lineItems.boughtAt.gte": date_from,
                "lineItems.boughtAt.lte": date_to,
                "limit": page_size,
                "offset": offset,
            }
            data = await self._get("/order/checkout-forms", params=params)
            page = [self._parse_order(o) for o in data.get("checkoutForms", [])]
            total_count = int(data.get("totalCount") or 0)
            all_fetched.extend(page)
            logger.info(
                "get_all_paid_orders_in_period: page offset=%d → %d orders (total_count=%d, running=%d)",
                offset, len(page), total_count, len(all_fetched),
            )
            offset += page_size
            if total_count and offset >= total_count:
                break
            if len(page) < page_size:
                break

        # Client-side filter by payment.finishedAt
        result = [o for o in all_fetched if date_from <= (o.paid_at or "") <= date_to]
        logger.info(
            "get_all_paid_orders_in_period: %d fetched → %d after paid_at filter (%s – %s)",
            len(all_fetched), len(result), date_from, date_to,
        )
        return result

    def _parse_order(self, data: dict) -> AllegroOrder:
        line_items = [
            AllegroOrderLine(
                offer_id=(item.get("offer") or {}).get("id", ""),
                offer_name=(item.get("offer") or {}).get("name", ""),
                quantity=item.get("quantity", 1),
                price=float((item.get("price") or {}).get("amount", 0) or 0),
                currency=(item.get("price") or {}).get("currency", "PLN"),
            )
            for item in data.get("lineItems", [])
        ]
        summary = data.get("summary") or {}
        total_amount = summary.get("totalToPay") or {}
        invoice_buyer = parse_invoice_buyer(data)
        invoice_required = invoice_buyer.required and not invoice_buyer.dont_want
        # delivery.time.dispatch = the window the seller has to hand the parcel
        # over to the carrier; `.to` is the deadline the store owner is held to.
        # Every level can legitimately be absent or null (older orders), so each
        # one is unwrapped defensively rather than chained.
        delivery = data.get("delivery") if isinstance(data.get("delivery"), dict) else {}
        delivery_time = delivery.get("time") if isinstance(delivery.get("time"), dict) else {}
        dispatch = delivery_time.get("dispatch") if isinstance(delivery_time.get("dispatch"), dict) else {}
        return AllegroOrder(
            order_id=data.get("id", ""),
            buyer_login=(data.get("buyer") or {}).get("login", ""),
            buyer_email=(data.get("buyer") or {}).get("email", ""),
            status=data.get("status", ""),
            fulfillment_status=(data.get("fulfillment") or {}).get("status", ""),
            payment_status=(data.get("payment") or {}).get("type", ""),
            paid_at=(data.get("payment") or {}).get("finishedAt", ""),
            total_price=float(total_amount.get("amount", 0) or 0) if isinstance(total_amount, dict) else 0.0,
            currency=total_amount.get("currency", "PLN") if isinstance(total_amount, dict) else "PLN",
            created_at=data.get("boughtAt", ""),
            dispatch_from=dispatch.get("from") or "",
            dispatch_to=dispatch.get("to") or "",
            delivery=delivery,
            line_items=line_items,
            invoice_required=invoice_required,
            invoice_buyer=invoice_buyer,
        )

    async def get_order_invoice_data(self, order_id: str) -> dict[str, Any]:
        """Return the full invoice address block from an order's checkout form.

        The dict shape (note the camelCase "dontWant") is what the invoicing flow
        and services/infakt_service.build_invoice_payload have always consumed;
        the parsing itself lives in parse_invoice_buyer, shared with _parse_order.
        """
        data = await self._get(f"/order/checkout-forms/{order_id}")
        buyer = parse_invoice_buyer(data)
        return {
            "required": buyer.required,
            "dontWant": buyer.dont_want,
            "company_name": buyer.company_name,
            "vat_id": buyer.vat_id,
            "first_name": buyer.first_name,
            "last_name": buyer.last_name,
            "street": buyer.street,
            "city": buyer.city,
            "zip_code": buyer.zip_code,
            "country_code": buyer.country_code,
        }

    async def get_order_invoices(self, order_id: str) -> list[dict[str, Any]]:
        cached = self._invoice_cache.get(order_id)
        if cached is not None:
            logger.debug("invoice cache hit: %s", order_id)
            return cached
        data = await self._get(f"/order/checkout-forms/{order_id}/invoices")
        invoices = data.get("invoices", [])
        self._invoice_cache.set(order_id, invoices)
        return invoices

    async def create_order_invoice_record(self, order_id: str, invoice_number: str, filename: str) -> str:
        """POST .../invoices — register invoice metadata on the order, return Allegro's invoice id.

        Must be followed by upload_order_invoice_file() with the same id.
        Needs the SCOPE_ORDERS_WRITE scope on the token.
        """
        body: dict[str, Any] = {"file": {"name": filename}}
        if invoice_number:
            body["invoiceNumber"] = invoice_number
        data = await self._post(f"/order/checkout-forms/{order_id}/invoices", body)
        self._invoice_cache.invalidate(order_id)
        return data["id"]

    async def upload_order_invoice_file(self, order_id: str, allegro_invoice_id: str, pdf_bytes: bytes) -> None:
        """PUT .../invoices/{invoiceId}/file — upload the actual PDF.

        Allegro takes up to 10 invoices per order, each at most
        INVOICE_FILE_MAX_BYTES. Needs the SCOPE_ORDERS_WRITE scope on the token;
        without it the upload comes back 403 and the seller is told they lack
        permission to add invoices.
        """
        await self._put_bytes(
            f"/order/checkout-forms/{order_id}/invoices/{allegro_invoice_id}/file",
            pdf_bytes,
            "application/pdf",
        )

    async def invoices_issued_map(self, order_ids: list[str]) -> dict[str, bool | None]:
        """order_id → True when a VAT invoice file is already attached to the
        order in Allegro, False when none is, None when the lookup failed.

        Allegro has no bulk endpoint for this — it is one GET per order — so the
        calls run at bounded concurrency instead of firing the whole list at the
        API at once (a year-long buyer report can ask about hundreds of orders).
        A single failed lookup degrades to None rather than sinking the whole
        report; the caller reports how many it could not check instead of
        quietly counting them as "no invoice".
        """
        semaphore = asyncio.Semaphore(_INVOICE_LOOKUP_CONCURRENCY)

        async def check(order_id: str) -> tuple[str, bool | None]:
            async with semaphore:
                try:
                    return order_id, bool(await self.get_order_invoices(order_id))
                except Exception as exc:  # noqa: BLE001 — one order must not sink the report
                    logger.warning("invoices_issued_map: lookup failed for %s: %s", order_id, exc)
                    return order_id, None

        return dict(await asyncio.gather(*[check(oid) for oid in order_ids]))

    async def get_orders_needing_invoice(
        self,
        month: int | None = None,
        year: int | None = None,
        shipped_only: bool = False,
    ) -> list[AllegroOrder]:
        """
        Return orders for the given month (default: current month) where:
          - buyer requested an invoice (invoice.required=true, dontWant=false)
          - seller hasn't uploaded one yet
          - if shipped_only: the order has already been sent to the buyer
            (fulfillment.status SENT or PICKED_UP) — used by the invoice
            reminder (services/invoice_reminder.py), which only nags about
            orders that already shipped, not ones still being packed.
        Paginates through all orders for the month, then checks invoice status.
        """
        import calendar
        from datetime import date

        today = date.today()
        m = month or today.month
        y = year or today.year
        first_day = date(y, m, 1).isoformat() + "T00:00:00Z"
        last_day = date(y, m, calendar.monthrange(y, m)[1]).isoformat() + "T23:59:59Z"

        # Paginate through all orders for the month
        all_orders: list[AllegroOrder] = []
        page_size = 50
        offset = 0
        while True:
            page = await self.get_orders(
                status="READY_FOR_PROCESSING",
                bought_at_gte=first_day,
                bought_at_lte=last_day,
                limit=page_size,
                offset=offset,
            )
            all_orders.extend(page)
            if len(page) < page_size:
                break
            offset += page_size

        # Client-side filter: buyer wants invoice
        candidates = [o for o in all_orders if o.invoice_required]
        if shipped_only:
            candidates = [o for o in candidates if o.fulfillment_status in ("SENT", "PICKED_UP")]

        # Keep only those without any uploaded invoice — fetch all in parallel
        invoice_lists = await asyncio.gather(*[
            self.get_order_invoices(o.order_id) for o in candidates
        ])
        return [o for o, invs in zip(candidates, invoice_lists) if not invs]

    # ── Offers ────────────────────────────────────────────────────────────────

    async def get_offers(
        self,
        publication_status: str = "ACTIVE",
        name: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> tuple[list[dict[str, Any]], int]:
        """Returns (offers, totalCount)."""
        params: dict[str, Any] = {
            "publication.status": publication_status,
            "limit": limit,
            "offset": offset,
        }
        if name:
            params["name"] = name
        data = await self._get("/sale/offers", params=params)
        offers = data.get("offers", [])
        # totalCount = all matching records, count = records in this page
        total = int(data.get("totalCount") or 0)
        return offers, total

    async def get_all_offers(self, publication_status: str = "ACTIVE") -> list[dict[str, Any]]:
        """Fetch every offer with pagination, parallelising after the first page. Cached for 5 min."""
        import asyncio as _asyncio

        cached = self._all_offers_cache.get(publication_status)
        if cached is not None:
            logger.info("get_all_offers: returning %d offers from cache", len(cached))
            return cached

        page_size = 100  # Allegro max per page

        # First page tells us totalCount so we can fan out the rest in parallel.
        first_page, total_count = await self.get_offers(
            publication_status=publication_status,
            limit=page_size,
            offset=0,
        )
        logger.info("get_all_offers: first page %d offers, totalCount=%s", len(first_page), total_count)

        all_offers = list(first_page)

        if total_count and total_count > len(first_page):
            offsets = range(page_size, total_count, page_size)
            pages = await _asyncio.gather(*[
                self.get_offers(
                    publication_status=publication_status,
                    limit=page_size,
                    offset=off,
                )
                for off in offsets
            ])
            for page, _ in pages:
                all_offers.extend(page)
            logger.info("get_all_offers: fetched %d parallel pages", len(pages))

        if total_count and len(all_offers) != total_count:
            logger.warning(
                "get_all_offers: fetched %d but API totalCount=%d (status=%s)",
                len(all_offers), total_count, publication_status,
            )

        logger.info("get_all_offers: done — %d offers total", len(all_offers))
        self._all_offers_cache.set(publication_status, all_offers)
        return all_offers

    async def get_offer(self, offer_id: str) -> dict[str, Any]:
        return await self._get(f"/sale/offers/{offer_id}")

    async def update_offer_price(self, offer_id: str, amount: float, currency: str = "PLN") -> dict:
        body = {
            "sellingMode": {
                "price": {"amount": str(amount), "currency": currency}
            }
        }
        return await self._post(f"/sale/offers/{offer_id}", body)

    async def update_offer_stock(self, offer_id: str, available: int) -> dict:
        body = {"stock": {"available": available}}
        return await self._post(f"/sale/offers/{offer_id}", body)

    # ── Messaging ─────────────────────────────────────────────────────────────

    async def get_message_threads(self, limit: int = 20) -> list[dict[str, Any]]:
        data = await self._get("/messaging/threads", params={"limit": min(limit, 20)})
        return data.get("threads", [])

    async def get_thread_messages(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        data = await self._get(f"/messaging/threads/{thread_id}/messages", params={"limit": min(limit, 20)})
        return data.get("messages", [])

    async def send_message(self, thread_id: str, text: str) -> dict[str, Any]:
        body = {"text": text, "type": "ANSWER"}
        return await self._post(f"/messaging/threads/{thread_id}/messages", body)

    async def create_thread(self, order_id: str, text: str) -> dict[str, Any]:
        body = {
            "order": {"id": order_id},
            "subject": {"id": "OTHER"},
            "message": {"text": text},
        }
        return await self._post("/messaging/threads", body)

    # ── User / Account ────────────────────────────────────────────────────────

    async def get_user_info(self) -> dict[str, Any]:
        return await self._get("/me")

    async def get_billing_entries(self, limit: int = 10) -> list[dict[str, Any]]:
        data = await self._get("/billing/billing-entries", params={"limit": limit})
        return data.get("billingEntries", [])

    async def get_billing_entries_for_order(self, order_id: str) -> list[dict[str, Any]]:
        """Fetch all billing entries for a specific order using order.id filter."""
        all_entries: list[dict[str, Any]] = []
        page_size = 100
        offset = 0
        while True:
            params = {"order.id": order_id, "limit": page_size, "offset": offset}
            data = await self._get("/billing/billing-entries", params=params)
            entries = data.get("billingEntries", [])
            logger.info(
                "get_billing_entries_for_order %s offset=%d: API returned %d entries, raw keys: %s",
                order_id, offset, len(entries),
                list(data.keys()),
            )
            for idx, e in enumerate(entries):
                logger.info(
                    "  billing[%d]: occurredAt=%s type=%s offer=%s amount=%s",
                    idx,
                    e.get("occurredAt", "")[:10],
                    (e.get("type") or {}).get("description", "?"),
                    (e.get("offer") or {}).get("name", "—"),
                    (e.get("value") or {}).get("amount", "?"),
                )
            all_entries.extend(entries)
            if len(entries) < page_size:
                break
            offset += page_size
        logger.info("get_billing_entries_for_order %s TOTAL: %d entries", order_id, len(all_entries))
        return all_entries

    @staticmethod
    def _split_period(date_from: str, date_to: str, window_days: int) -> list[tuple[str, str]]:
        """Cut [date_from, date_to] into windows of at most `window_days`.

        Windows share their boundary instant (both bounds are inclusive on the
        Allegro side), so nothing that occurred exactly on a boundary can slip
        between two windows; callers de-duplicate by entry id.
        """
        fmt = "%Y-%m-%dT%H:%M:%SZ"
        start = datetime.fromisoformat(date_from.replace("Z", "+00:00"))
        end = datetime.fromisoformat(date_to.replace("Z", "+00:00"))
        if end <= start:
            return [(date_from, date_to)]
        windows: list[tuple[str, str]] = []
        cursor = start
        step = timedelta(days=window_days)
        while cursor < end:
            window_end = min(cursor + step, end)
            windows.append((cursor.strftime(fmt), window_end.strftime(fmt)))
            cursor = window_end
        return windows

    async def _get_with_retry(
        self, path: str, params: dict | list | None = None, attempts: int = 3
    ) -> dict[str, Any]:
        """GET that retries the transient failures (429, 5xx, network) with backoff.

        Anything else — a 403 for a missing scope, a 400 for a bad filter — is
        raised on the first try: retrying it would only make the caller wait.
        """
        for attempt in range(1, attempts + 1):
            try:
                return await self._get(path, params=params)
            except AllegroAPIError as exc:
                if attempt == attempts or exc.status_code not in _RETRYABLE_STATUSES:
                    raise
                delay = 2.0 ** (attempt - 1)
                logger.warning(
                    "GET %s failed (%s), retry %d/%d in %.0fs",
                    path, exc, attempt, attempts - 1, delay,
                )
                await asyncio.sleep(delay)
        raise AssertionError("unreachable")  # pragma: no cover

    async def _get_billing_window(self, date_from: str, date_to: str) -> list[dict[str, Any]]:
        """All billing entries of one window, paginated."""
        entries: list[dict[str, Any]] = []
        page_size = 100
        for page in range(_BILLING_MAX_PAGES):
            params = {
                "occurredAt.gte": date_from,
                "occurredAt.lte": date_to,
                "limit": page_size,
                "offset": page * page_size,
            }
            data = await self._get_with_retry("/billing/billing-entries", params=params)
            page_entries = data.get("billingEntries", [])
            entries.extend(page_entries)
            if len(page_entries) < page_size:
                return entries
        logger.warning(
            "get_billing_entries_in_period: window %s – %s hit the %d-page cap (%d entries) — "
            "cost data for it may be incomplete",
            date_from[:10], date_to[:10], _BILLING_MAX_PAGES, len(entries),
        )
        return entries

    async def get_billing_entries_in_period(self, date_from: str, date_to: str) -> list[dict[str, Any]]:
        """Fetch all billing entries in a date range (windowed + paginated).

        Requires the `allegro:api:billing:read` scope on the token; without it
        Allegro answers 403 and the caller has to say so rather than report
        zero costs.
        """
        windows = self._split_period(date_from, date_to, _BILLING_WINDOW_DAYS)
        all_entries: list[dict[str, Any]] = []
        seen_ids: set[str] = set()
        for window_from, window_to in windows:
            entries = await self._get_billing_window(window_from, window_to)
            new = 0
            for entry in entries:
                entry_id = entry.get("id")
                if entry_id:
                    if entry_id in seen_ids:
                        continue
                    seen_ids.add(entry_id)
                all_entries.append(entry)
                new += 1
            logger.info(
                "get_billing_entries_in_period: window %s – %s → %d entries (%d new, total: %d)",
                window_from[:10], window_to[:10], len(entries), new, len(all_entries),
            )
        logger.info(
            "get_billing_entries_in_period: %d window(s) %s – %s → %d entries",
            len(windows), date_from[:10], date_to[:10], len(all_entries),
        )
        return all_entries

    async def get_carriers(self) -> list[dict[str, Any]]:
        """Return list of available Allegro carriers (id + name)."""
        data = await self._get("/order/carriers")
        return data.get("carriers", [])

    async def get_order_event_stats(self) -> dict[str, Any]:
        """Return the latest event ID and timestamp — use for monitoring baseline."""
        data = await self._get("/order/event-stats")
        latest = data.get("latestEvent") or {}
        return {"latest_event_id": latest.get("id"), "occurred_at": latest.get("occurredAt")}

    @staticmethod
    def order_event_details(order: AllegroOrder) -> dict[str, Any]:
        """The order facts a new-order notification carries beyond its ID.

        The monitor announces a new order without the seller having to open
        anything, so the push has to answer what they always ask next: how much
        is it worth, how does it ship, and does it need an invoice.
        """
        delivery = order.delivery if isinstance(order.delivery, dict) else {}
        method = delivery.get("method") if isinstance(delivery.get("method"), dict) else {}
        return {
            "total_price": order.total_price,
            "currency": order.currency,
            "delivery_method": method.get("name") or "",
            "invoice_required": order.invoice_required,
        }

    async def get_order_events(self, since_event_id: str | None = None) -> dict[str, Any]:
        """Fetch new READY_FOR_PROCESSING order events since a given event ID.

        Verifies fulfillment.status == NEW to avoid false positives from orders
        that were cancelled and re-paid (payment event fires again but order is SENT).
        Each returned order also carries the details the push notification shows
        (value, delivery method, invoice flag) — they come from the very
        checkout-form fetched for that verification, so they cost no extra call.
        """
        import asyncio

        params_list: list[tuple[str, str]] = [("type[]", "READY_FOR_PROCESSING"), ("limit", "100")]
        if since_event_id:
            params_list.append(("from", since_event_id))
        data = await self._get("/order/events", params=params_list)
        events = data.get("events", [])
        last_event_id = events[-1]["id"] if events else since_event_id

        candidates = [
            {
                "event_id": e["id"],
                "order_id": ((e.get("order") or {}).get("checkoutForm") or {}).get("id"),
                "occurred_at": e.get("occurredAt"),
            }
            for e in events
            if e.get("type") == "READY_FOR_PROCESSING"
        ]

        # Verify fulfillment.status to filter out false positives (e.g. re-paid cancelled orders)
        if candidates:
            order_results = await asyncio.gather(
                *[self.get_order(c["order_id"]) for c in candidates if c["order_id"]],
                return_exceptions=True,
            )
            order_map: dict[str, AllegroOrder] = {}
            for result in order_results:
                if isinstance(result, BaseException):
                    continue
                order_map[result.order_id] = result
            kept: list[dict[str, Any]] = []
            for c in candidates:
                order = order_map.get(c["order_id"])
                # An order whose fetch failed stays in (unknown status is treated
                # as NEW, as before) — it just goes out without the extra details.
                if order is not None:
                    if order.fulfillment_status != "NEW":
                        continue
                    c.update(self.order_event_details(order))
                kept.append(c)
            candidates = kept

        return {
            "new_orders": candidates,
            "last_event_id": last_event_id,
            "count": len(candidates),
        }

    # ── Returns & complaints ─────────────────────────────────────────────────

    # /order/customer-returns is still a beta resource — the default
    # public.v1 Accept header gets a 406 NotAcceptableException.
    _RETURNS_ACCEPT = "application/vnd.allegro.beta.v1+json"

    # One period query must never be answered from a single page: "ile zwrotów
    # w tym miesiącu" has to count every return in the window, not the first
    # page of them (that bug reported the page size, 50, as the count).
    _RETURNS_PAGE_SIZE = 50
    _RETURNS_MAX_PAGES = 20

    async def get_customer_returns(
        self,
        limit: int = 50,
        status: str | None = None,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return recent customer returns (zwroty) — GET /order/customer-returns.

        `status`, if given, filters server-side (e.g. status="DELIVERED" for
        returns whose parcel has arrived back and is awaiting a seller
        decision — accept/refund or reject).

        `date_from`/`date_to` (UTC ISO 8601, both required together) restrict
        the result to returns created in that window. The window is paged
        through to the end and filtered client-side on createdAt as well, so
        the returned list is the COMPLETE set for the period — callers can
        count it. Without a period the call stays a single `limit`-sized page,
        as the returns monitor and the plain "nowe zwroty" listing want.

        Newest first isn't guaranteed by the API, so callers that need "most
        recent" should sort client-side by whatever date field is present.
        """
        if not (date_from and date_to):
            params: dict[str, Any] = {"limit": limit}
            if status:
                params["status"] = status
            data = await self._get("/order/customer-returns", params=params, accept=self._RETURNS_ACCEPT)
            return data.get("customerReturns", [])

        base: dict[str, Any] = {"status": status} if status else {}
        try:
            fetched = await self._fetch_customer_return_pages(
                {**base, "createdAt.gte": date_from, "createdAt.lte": date_to}
            )
        except AllegroAPIError as exc:
            if exc.status_code != 400:
                raise
            # customer-returns is a beta resource; if it rejects the createdAt
            # filters, page through unfiltered and rely on the client-side
            # filter below rather than failing the whole query.
            logger.warning(
                "get_customer_returns: createdAt filter rejected (400), falling back to client-side filtering"
            )
            fetched = await self._fetch_customer_return_pages(base)

        result = [
            item for item in fetched
            if _in_utc_period(return_created_at(item), date_from, date_to)
        ]
        logger.info(
            "get_customer_returns: %d fetched → %d in period %s – %s (status=%s)",
            len(fetched), len(result), date_from, date_to, status or "any",
        )
        return result

    async def _fetch_customer_return_pages(self, params: dict[str, Any]) -> list[dict[str, Any]]:
        """Page through /order/customer-returns with `params` until exhausted."""
        collected: list[dict[str, Any]] = []
        offset = 0
        for _ in range(self._RETURNS_MAX_PAGES):
            data = await self._get(
                "/order/customer-returns",
                params={**params, "limit": self._RETURNS_PAGE_SIZE, "offset": offset},
                accept=self._RETURNS_ACCEPT,
            )
            page = data.get("customerReturns", [])
            collected.extend(page)
            if len(page) < self._RETURNS_PAGE_SIZE:
                break
            offset += self._RETURNS_PAGE_SIZE
        else:
            logger.warning(
                "_fetch_customer_return_pages: hit the %d-page cap — result may be truncated",
                self._RETURNS_MAX_PAGES,
            )
        return collected

    async def get_customer_return(self, return_id: str) -> dict[str, Any]:
        return await self._get(f"/order/customer-returns/{return_id}", accept=self._RETURNS_ACCEPT)

    # /sale/issues is the beta.v1 successor to the removed /sale/disputes —
    # covers both buyer disputes and formal claims (reklamacje).
    _ISSUES_ACCEPT = "application/vnd.allegro.beta.v1+json"

    async def get_issues(
        self,
        limit: int = 50,
        date_from: str | None = None,
        date_to: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return open disputes/claims (reklamacje) — GET /sale/issues.

        `date_from`/`date_to` (UTC ISO 8601, both required together) restrict
        the result to issues opened in that window. /sale/issues exposes no
        documented date filter, so the window is applied client-side over the
        full paged listing — same contract as get_customer_returns: the list
        is the COMPLETE set for the period and can be counted.
        """
        if not (date_from and date_to):
            data = await self._get("/sale/issues", params={"limit": limit}, accept=self._ISSUES_ACCEPT)
            return data.get("issues", [])

        fetched: list[dict[str, Any]] = []
        offset = 0
        for _ in range(self._RETURNS_MAX_PAGES):
            data = await self._get(
                "/sale/issues",
                params={"limit": self._RETURNS_PAGE_SIZE, "offset": offset},
                accept=self._ISSUES_ACCEPT,
            )
            page = data.get("issues", [])
            fetched.extend(page)
            if len(page) < self._RETURNS_PAGE_SIZE:
                break
            offset += self._RETURNS_PAGE_SIZE
        else:
            logger.warning(
                "get_issues: hit the %d-page cap — result may be truncated", self._RETURNS_MAX_PAGES
            )

        result = [
            item for item in fetched
            if _in_utc_period(issue_opened_at(item), date_from, date_to)
        ]
        logger.info(
            "get_issues: %d fetched → %d in period %s – %s", len(fetched), len(result), date_from, date_to
        )
        return result

    async def get_issue(self, issue_id: str) -> dict[str, Any]:
        return await self._get(f"/sale/issues/{issue_id}", accept=self._ISSUES_ACCEPT)

    async def close(self) -> None:
        await self._client.aclose()
