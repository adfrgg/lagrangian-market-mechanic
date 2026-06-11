"""Order book primitives for L2 Lagrangian Market Mechanics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np


Level = tuple[float, float]


@dataclass
class OrderBook:
    """Small research order book with price-level update support."""

    bids: dict[float, float] = field(default_factory=dict)
    asks: dict[float, float] = field(default_factory=dict)

    @classmethod
    def from_snapshot(cls, bids: Iterable[Level], asks: Iterable[Level]) -> "OrderBook":
        book = cls()
        for price, qty in bids:
            book.update_bid(price, qty)
        for price, qty in asks:
            book.update_ask(price, qty)
        return book

    def update_bid(self, price: float, qty: float) -> None:
        self._update(self.bids, price, qty)

    def update_ask(self, price: float, qty: float) -> None:
        self._update(self.asks, price, qty)

    def apply_snapshot(self, bids: Iterable[Level], asks: Iterable[Level]) -> None:
        self.bids.clear()
        self.asks.clear()
        for price, qty in bids:
            self.update_bid(price, qty)
        for price, qty in asks:
            self.update_ask(price, qty)

    def apply_updates(self, bids: Iterable[Level] = (), asks: Iterable[Level] = ()) -> None:
        for price, qty in bids:
            self.update_bid(price, qty)
        for price, qty in asks:
            self.update_ask(price, qty)

    def best_bid(self) -> float:
        return max(self.bids) if self.bids else np.nan

    def best_ask(self) -> float:
        return min(self.asks) if self.asks else np.nan

    def mid_price(self) -> float:
        bid = self.best_bid()
        ask = self.best_ask()
        return 0.5 * (bid + ask) if np.isfinite(bid) and np.isfinite(ask) else np.nan

    def spread(self) -> float:
        bid = self.best_bid()
        ask = self.best_ask()
        return ask - bid if np.isfinite(bid) and np.isfinite(ask) else np.nan

    def top_n_bids(self, n: int) -> list[Level]:
        return sorted(self.bids.items(), key=lambda item: item[0], reverse=True)[:n]

    def top_n_asks(self, n: int) -> list[Level]:
        return sorted(self.asks.items(), key=lambda item: item[0])[:n]

    def bid_depth_quote(self, n: int) -> float:
        return float(sum(price * qty for price, qty in self.top_n_bids(n)))

    def ask_depth_quote(self, n: int) -> float:
        return float(sum(price * qty for price, qty in self.top_n_asks(n)))

    def book_imbalance(self, n: int, eps: float = 1e-12) -> float:
        bid_depth = self.bid_depth_quote(n)
        ask_depth = self.ask_depth_quote(n)
        return float((bid_depth - ask_depth) / (bid_depth + ask_depth + eps))

    def cost_to_move_up_bps(self, horizon_bps: float) -> float:
        mid = self.mid_price()
        if not np.isfinite(mid):
            return np.nan
        target = mid * math.exp(horizon_bps / 10000.0)
        levels = self.top_n_asks(len(self.asks))
        if not levels or levels[-1][0] < target:
            return np.nan
        return float(sum(price * qty for price, qty in levels if price <= target))

    def cost_to_move_down_bps(self, horizon_bps: float) -> float:
        mid = self.mid_price()
        if not np.isfinite(mid):
            return np.nan
        target = mid * math.exp(-horizon_bps / 10000.0)
        levels = self.top_n_bids(len(self.bids))
        if not levels or levels[-1][0] > target:
            return np.nan
        return float(sum(price * qty for price, qty in levels if price >= target))

    def insufficient_depth_up(self, horizon_bps: float) -> bool:
        return not np.isfinite(self.cost_to_move_up_bps(horizon_bps))

    def insufficient_depth_down(self, horizon_bps: float) -> bool:
        return not np.isfinite(self.cost_to_move_down_bps(horizon_bps))

    def snapshot_row(self, n_levels: int = 20) -> dict[str, float]:
        row: dict[str, float] = {
            "best_bid": self.best_bid(),
            "best_ask": self.best_ask(),
            "mid": self.mid_price(),
            "spread": self.spread(),
        }
        bids = self.top_n_bids(n_levels)
        asks = self.top_n_asks(n_levels)
        for i in range(1, n_levels + 1):
            bid = bids[i - 1] if i <= len(bids) else (np.nan, np.nan)
            ask = asks[i - 1] if i <= len(asks) else (np.nan, np.nan)
            row[f"bid_price_{i}"] = bid[0]
            row[f"bid_qty_{i}"] = bid[1]
            row[f"ask_price_{i}"] = ask[0]
            row[f"ask_qty_{i}"] = ask[1]
        return row

    @staticmethod
    def _update(side: dict[float, float], price: float, qty: float) -> None:
        price = float(price)
        qty = float(qty)
        if qty == 0.0:
            side.pop(price, None)
        elif qty > 0.0 and np.isfinite(price):
            side[price] = qty
