from __future__ import annotations

from finantradealgo.system.config_loader import (
    load_system_config,
    load_exchange_credentials,
)
from finantradealgo.execution.exchange_client import BinanceFuturesClient


def main() -> None:
    cfg = load_system_config()
    exchange_cfg = cfg.get("exchange_cfg")
    if exchange_cfg is None:
        raise RuntimeError("Exchange config not found. Check config/system.yml.")

    api_key, secret = load_exchange_credentials(exchange_cfg)
    client = BinanceFuturesClient(exchange_cfg, api_key, secret)

    account = client.get_account_info()
    print("Account info:", account)

    symbol = cfg.get("symbol", "AIAUSDT")
    exch_symbol = exchange_cfg.symbol_mapping.get(symbol, symbol)

    if getattr(exchange_cfg, "dry_run", True):
        print(
            "Dry run enabled (exchange.dry_run=true). Order placement skipped. "
            "Set dry_run=false in config/system.yml to test actual order flow."
        )
        return

    order = client.place_order(
        symbol=exch_symbol,
        side="BUY",
        qty=0.001,
        order_type="LIMIT",
        price=1.0,
        timeInForce="GTC",
    )
    print("Order placed:", order)

    client.cancel_order(exch_symbol, order["orderId"])
    print("Order cancelled.")


if __name__ == "__main__":
    main()
