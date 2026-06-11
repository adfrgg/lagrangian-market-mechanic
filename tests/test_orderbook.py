import numpy as np

from lagrangian_market.orderbook import OrderBook


def test_orderbook_snapshot_sorting_and_basic_prices() -> None:
    book = OrderBook.from_snapshot(
        bids=[(99.0, 2.0), (100.0, 1.0)],
        asks=[(102.0, 1.5), (101.0, 1.0)],
    )

    assert book.top_n_bids(2) == [(100.0, 1.0), (99.0, 2.0)]
    assert book.top_n_asks(2) == [(101.0, 1.0), (102.0, 1.5)]
    assert book.best_bid() == 100.0
    assert book.best_ask() == 101.0
    assert book.mid_price() == 100.5
    assert book.spread() == 1.0


def test_orderbook_update_insert_change_remove() -> None:
    book = OrderBook.from_snapshot(bids=[(100.0, 1.0)], asks=[(101.0, 1.0)])
    book.update_bid(100.0, 2.0)
    book.update_bid(99.5, 3.0)
    book.update_ask(101.0, 0.0)
    book.update_ask(100.8, 1.2)

    assert book.bids[100.0] == 2.0
    assert book.best_ask() == 100.8
    assert 101.0 not in book.asks


def test_depth_imbalance_and_cost_to_move() -> None:
    book = OrderBook.from_snapshot(
        bids=[(100.0, 1.0), (99.9, 2.0), (99.8, 3.0)],
        asks=[(100.1, 1.0), (100.2, 2.0), (100.3, 3.0)],
    )

    assert book.bid_depth_quote(2) == 100.0 * 1.0 + 99.9 * 2.0
    assert book.ask_depth_quote(2) == 100.1 * 1.0 + 100.2 * 2.0
    assert np.isfinite(book.book_imbalance(2))
    assert np.isfinite(book.cost_to_move_up_bps(10))
    assert np.isfinite(book.cost_to_move_down_bps(10))
    assert np.isnan(book.cost_to_move_up_bps(1000))
