from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List, Any
import string

import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
import pandas as pd
import numpy as np
import jsonpickle
import statistics
import math
from typing import List, Dict

class Logger:
    def __init__(self) -> None:
        self.logs = ""
        self.max_log_length = 3750

    def print(self, *objects: Any, sep: str = " ", end: str = "\n") -> None:
        self.logs += sep.join(map(str, objects)) + end

    def flush(self, state: TradingState, orders: dict[Symbol, list[Order]], conversions: int, trader_data: str) -> None:
        base_length = len(self.to_json([
            self.compress_state(state, ""),
            self.compress_orders(orders),
            conversions,
            "",
            "",
        ]))

        # We truncate state.traderData, trader_data, and self.logs to the same max. length to fit the log limit
        max_item_length = (self.max_log_length - base_length) // 3

        print(self.to_json([
            self.compress_state(state, self.truncate(state.traderData, max_item_length)),
            self.compress_orders(orders),
            conversions,
            self.truncate(trader_data, max_item_length),
            self.truncate(self.logs, max_item_length),
        ]))

        self.logs = ""

    def compress_state(self, state: TradingState, trader_data: str) -> list[Any]:
        return [
            state.timestamp,
            trader_data,
            self.compress_listings(state.listings),
            self.compress_order_depths(state.order_depths),
            self.compress_trades(state.own_trades),
            self.compress_trades(state.market_trades),
            state.position,
            self.compress_observations(state.observations),
        ]

    def compress_listings(self, listings: dict[Symbol, Listing]) -> list[list[Any]]:
        compressed = []
        for listing in listings.values():
            compressed.append([listing["symbol"], listing["product"], listing["denomination"]])

        return compressed

    def compress_order_depths(self, order_depths: dict[Symbol, OrderDepth]) -> dict[Symbol, list[Any]]:
        compressed = {}
        for symbol, order_depth in order_depths.items():
            compressed[symbol] = [order_depth.buy_orders, order_depth.sell_orders]

        return compressed

    def compress_trades(self, trades: dict[Symbol, list[Trade]]) -> list[list[Any]]:
        compressed = []
        for arr in trades.values():
            for trade in arr:
                compressed.append([
                    trade.symbol,
                    trade.price,
                    trade.quantity,
                    trade.buyer,
                    trade.seller,
                    trade.timestamp,
                ])

        return compressed

    def compress_observations(self, observations: Observation) -> list[Any]:
        conversion_observations = {}
        for product, observation in observations.conversionObservations.items():
            conversion_observations[product] = [
                observation.bidPrice,
                observation.askPrice,
                observation.transportFees,
                observation.exportTariff,
                observation.importTariff,
                observation.sunlight,
                observation.humidity,
            ]

        return [observations.plainValueObservations, conversion_observations]

    def compress_orders(self, orders: dict[Symbol, list[Order]]) -> list[list[Any]]:
        compressed = []
        for arr in orders.values():
            for order in arr:
                compressed.append([order.symbol, order.price, order.quantity])

        return compressed

    def to_json(self, value: Any) -> str:
        return json.dumps(value, cls=ProsperityEncoder, separators=(",", ":"))

    def truncate(self, value: str, max_length: int) -> str:
        if len(value) <= max_length:
            return value

        return value[:max_length - 3] + "..."

logger = Logger()
class Trader:

    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}

    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        logger.print("traderData: " + state.traderData)
        logger.print("Observations: " + str(state.observations))
        
        result = {}
        
        for product in state.order_depths:
            order_depth: OrderDepth = state.order_depths[product]
            orders: List[Order] = []
            
            if product == "AMETHYSTS":
                buy_price = 9998  # Adjusted buy price
                sell_price = 10002  # Adjusted sell price
            elif product == "STARFRUIT":
                # Use simple moving average to predict future prices
                recent_trades = state.market_trades[product][-6:] if product in state.market_trades else []
                if recent_trades:
                    predicted_price = sum(trade.price for trade in recent_trades) / len(recent_trades)
                    # Set the buy and sell prices based on the predicted price with tighter spread
                    buy_price = int(predicted_price * 0.9991)  # 0.1% lower than predicted
                    sell_price = int(predicted_price * 1.0009)  # 0.1% higher than predicted
                    # Adjust sell price to ensure no loss
                    if sell_price <= buy_price:
                        sell_price = buy_price + 1
                    # Increase sell price intraday
                    sell_price += state.timestamp % 100
                    # Check for trough in the graph
                    if len(recent_trades) >= 2 and recent_trades[-1].price < recent_trades[-2].price:
                        # If there is a trough, increase the buy price to purchase more
                        buy_price = int(predicted_price * 0.99955)  # 0.055% lower than predicted
                    
                    buy_price = int(buy_price * 1.00055)  # Increase buy price by 0.02%
                else:
                    # If no recent trades, use default prices
                    buy_price = 4960
                    sell_price = 5060
            else:
                continue
            
            logger.print("Buy price : " + str(buy_price))
            logger.print("Sell price : " + str(sell_price))
            logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))
            
            current_position = state.position.get(product, 0)
            logger.print("Current position: " + str(current_position))
            
            # Market-making strategy
            if current_position < self.POSITION_LIMIT[product]:
                if len(order_depth.sell_orders) != 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if int(best_ask) <= buy_price:
                        quantity = self.POSITION_LIMIT[product] - current_position
                        logger.print("BUY", str(best_ask_amount) + "x", best_ask)
                        orders.append(Order(product, best_ask, quantity))
            
            if current_position > 0:
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    if int(best_bid) >= sell_price:
                        quantity = current_position
                        logger.print("SELL", str(best_bid_amount) + "x", best_bid)
                        orders.append(Order(product, best_bid, -quantity))
            
            # Market-taking strategy
            if current_position < self.POSITION_LIMIT[product]:
                if len(order_depth.sell_orders) != 0:
                    best_ask = min(order_depth.sell_orders.keys())
                    best_ask_amount = order_depth.sell_orders[best_ask]
                    if int(best_ask) <= buy_price:
                        quantity = min(best_ask_amount, self.POSITION_LIMIT[product] - current_position)
                        logger.print("BUY (Market Taker)", str(quantity) + "x", best_ask)
                        orders.append(Order(product, best_ask, quantity))
            
            if current_position > 0:
                if len(order_depth.buy_orders) != 0:
                    best_bid = max(order_depth.buy_orders.keys())
                    best_bid_amount = order_depth.buy_orders[best_bid]
                    if int(best_bid) >= sell_price:
                        quantity = min(best_bid_amount, current_position)
                        logger.print("SELL (Market Taker)", str(quantity) + "x", best_bid)
                        orders.append(Order(product, best_bid, -quantity))
            
            result[product] = orders
        
        traderData = "SAMPLE"  # String value holding Trader state data required. It will be delivered as TradingState.traderData on next execution.
        conversions = 1
        logger.flush(state, result, conversions, traderData)
        return result, conversions, traderData