from __future__ import annotations

"""Allegro REST API client with OAuth2 device-flow authentication."""

import json
import logging
from datetime import datetime, timedelta
from pathlib import Path
from time import monotonic
from typing import Any

import httpx

from config.settings import get_settings
from models.allegro import AllegroOrder, AllegroOrderLine, AllegroTokens

logger = logging.getLogger(__name__)


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


class AllegroAuthError(Exception):
    pass


async def exchange_allegro_code(code: str) -> tuple[str, "AllegroTokens"]:
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
                "redirect_uri": settings.allegro_redirect_uri,
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
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        super().__init__(f"Allegro API error {status_code}: {detail}")


class AllegroService:
    """
    Wraps the Allegro REST API.

    Authentication uses OAuth2 device flow (suitable for server-side apps).
    Tokens are refreshed automatically before expiry.
    Token persistence: Redis when REDIS_URL is set (survives redeployments),
    otherwise local file fallback.
    """

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
        # Static order fields (items, price, buyer) don't change — 5 min TTL
        self._order_cache: _TTLCache = _TTLCache(ttl=300.0)
        # Order list results — 60 s TTL (new orders can arrive)
        self._orders_list_cache: _TTLCache = _TTLCache(ttl=60.0)
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
        # File — fast local cache (ephemeral, but useful within a single deployment)
        try:
            self._token_file().write_text(json.dumps(data, indent=2))
        except Exception as exc:
            logger.warning("Could not write token file: %s", exc)
        # Redis — persists across Railway redeployments
        if self._redis is not None:
            try:
                await self._redis.set(self._redis_tokens_key, json.dumps(data))
                logger.info("Saved Allegro tokens to Redis")
            except Exception as exc:
                logger.warning("Failed to save Allegro tokens to Redis: %s", exc)

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
                    "scope": "allegro:api:sale:offers:read allegro:api:orders:read "
                             "allegro:api:orders:write allegro:api:messaging",
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

    async def _get(self, path: str, params: dict | None = None) -> dict[str, Any]:
        headers = await self._get_headers()
        resp = await self._client.get(path, headers=headers, params=params)
        if resp.status_code >= 400:
            raise AllegroAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    async def _post(self, path: str, body: dict) -> dict[str, Any]:
        headers = await self._get_headers()
        resp = await self._client.post(path, headers=headers, json=body)
        if resp.status_code >= 400:
            raise AllegroAPIError(resp.status_code, resp.text[:500])
        return resp.json()

    # ── Orders ────────────────────────────────────────────────────────────────

    async def get_orders(
        self,
        status: str | None = None,
        buyer_login: str | None = None,
        fulfillment_status: str | None = None,
        line_items_sent: list[str] | None = None,
        bought_at_gte: str | None = None,
        bought_at_lte: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[AllegroOrder]:
        cache_key = (
            f"{status}:{buyer_login}:{fulfillment_status}:{line_items_sent}:"
            f"{bought_at_gte}:{bought_at_lte}:{limit}:{offset}"
        )
        cached = self._orders_list_cache.get(cache_key)
        if cached is not None:
            logger.debug("orders list cache hit: %s", cache_key)
            return cached

        params: dict[str, Any] = {"limit": limit, "offset": offset}
        if status:
            params["status"] = status
        if buyer_login:
            params["buyer.login"] = buyer_login
        if fulfillment_status:
            params["fulfillment.status"] = fulfillment_status
        if line_items_sent:
            params["fulfillment.shipmentSummary.lineItemsSent"] = line_items_sent
        if bought_at_gte:
            params["lineItems.boughtAt.gte"] = bought_at_gte
        if bought_at_lte:
            params["lineItems.boughtAt.lte"] = bought_at_lte
        data = await self._get("/order/checkout-forms", params=params)
        orders = [self._parse_order(o) for o in data.get("checkoutForms", [])]
        self._orders_list_cache.set(cache_key, orders)
        for order in orders:
            self._order_cache.set(order.order_id, order)
        return orders

    async def get_order(self, order_id: str) -> AllegroOrder:
        cached = self._order_cache.get(order_id)
        if cached is not None:
            logger.debug("order cache hit: %s", order_id)
            return cached
        data = await self._get(f"/order/checkout-forms/{order_id}")
        order = self._parse_order(data)
        self._order_cache.set(order_id, order)
        return order

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
        invoice = data.get("invoice") or {}
        invoice_required = bool(invoice.get("required")) and not bool(invoice.get("dontWant"))
        return AllegroOrder(
            order_id=data.get("id", ""),
            buyer_login=(data.get("buyer") or {}).get("login", ""),
            buyer_email=(data.get("buyer") or {}).get("email", ""),
            status=data.get("status", ""),
            fulfillment_status=(data.get("fulfillment") or {}).get("status", ""),
            payment_status=((data.get("payment") or {}).get("paidAmount") or {}).get("currency", ""),
            total_price=float(total_amount.get("amount", 0) or 0) if isinstance(total_amount, dict) else 0.0,
            currency=total_amount.get("currency", "PLN") if isinstance(total_amount, dict) else "PLN",
            created_at=data.get("boughtAt", ""),
            delivery=data.get("delivery") or {},
            line_items=line_items,
            invoice_required=invoice_required,
        )

    async def get_order_invoices(self, order_id: str) -> list[dict[str, Any]]:
        cached = self._invoice_cache.get(order_id)
        if cached is not None:
            logger.debug("invoice cache hit: %s", order_id)
            return cached
        data = await self._get(f"/order/checkout-forms/{order_id}/invoices")
        invoices = data.get("invoices", [])
        self._invoice_cache.set(order_id, invoices)
        return invoices

    async def get_orders_needing_invoice(
        self,
        month: int | None = None,
        year: int | None = None,
    ) -> list[AllegroOrder]:
        """
        Return orders for the given month (default: current month) where:
          - buyer requested an invoice (invoice.required=true, dontWant=false)
          - seller hasn't uploaded one yet
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

        # Keep only those without any uploaded invoice
        result = []
        for order in candidates:
            invoices = await self.get_order_invoices(order.order_id)
            if not invoices:
                result.append(order)
        return result

    # ── Offers ────────────────────────────────────────────────────────────────

    async def get_offers(
        self,
        publication_status: str = "ACTIVE",
        name: str | None = None,
        limit: int = 20,
        offset: int = 0,
    ) -> list[dict[str, Any]]:
        params: dict[str, Any] = {
            "publication.status": publication_status,
            "limit": limit,
            "offset": offset,
        }
        if name:
            params["name"] = name
        data = await self._get("/sale/offers", params=params)
        return data.get("offers", [])

    async def get_all_offers(self, publication_status: str = "ACTIVE") -> list[dict[str, Any]]:
        """Fetch every offer with pagination. Cached for 5 minutes."""
        cached = self._all_offers_cache.get(publication_status)
        if cached is not None:
            return cached
        all_offers: list[dict[str, Any]] = []
        offset = 0
        page_size = 50
        while True:
            page = await self.get_offers(
                publication_status=publication_status,
                limit=page_size,
                offset=offset,
            )
            all_offers.extend(page)
            if len(page) < page_size:
                break
            offset += page_size
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
        data = await self._get("/messaging/threads", params={"limit": limit})
        return data.get("threads", [])

    async def get_thread_messages(self, thread_id: str, limit: int = 20) -> list[dict[str, Any]]:
        data = await self._get(f"/messaging/threads/{thread_id}/messages", params={"limit": limit})
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

    async def close(self) -> None:
        await self._client.aclose()
