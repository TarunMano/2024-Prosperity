from datamodel import OrderDepth, UserId, TradingState, Order
from typing import List
import string

import json
from datamodel import Listing, Observation, Order, OrderDepth, ProsperityEncoder, Symbol, Trade, TradingState
from typing import Any
import numpy as np
import pandas as pd

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

class Trader:
    POSITION_LIMIT = {'AMETHYSTS': 20, 'STARFRUIT': 20}
    price_files = [
        '/Users/prestonbui/Downloads/round-1-island-data-bottle (1)/prices_round_1_day_-2.csv',
        '/Users/prestonbui/Downloads/round-1-island-data-bottle (1)/prices_round_1_day_-1.csv',
        '/Users/prestonbui/Downloads/round-1-island-data-bottle (1)/prices_round_1_day_0.csv',
    ]
    trade_files = [
        '/Users/prestonbui/Downloads/round-1-island-data-bottle (1)/trades_round_1_day_-2_nn.csv',
        '/Users/prestonbui/Downloads/round-1-island-data-bottle (1)/trades_round_1_day_-1_nn.csv',
        '/Users/prestonbui/Downloads/round-1-island-data-bottle (1)/trades_round_1_day_0_nn.csv',
    ]


    def __init__(self, price_files=None, trade_files=None):
        self.logger = Logger()
        self.price_files = price_files if price_files is not None else []
        self.trade_files = trade_files if trade_files is not None else []
        self.historical_prices_cache = {}
        # Add moving average for STARFRUIT.
        self.moving_average_starfruit = []


    def read_price_data(self, file_paths, product_name):
        """
        This method is adjusted to correctly handle CSV files with different separators.
        It caches the price data for efficiency.
        """
        if product_name in self.historical_prices_cache:
            return self.historical_prices_cache[product_name]
        
        all_prices = []
        for file_path in file_paths:
            df = pd.read_csv(file_path, delimiter=';', skipinitialspace=True)
            if product_name in df['product'].values:
                prices = df[df['product'] == product_name]['mid_price'].tolist()
                all_prices.extend(prices)
            else:
                self.logger.print(f"'product' column or product name {product_name} not found in file: {file_path}")
        
        self.historical_prices_cache[product_name] = all_prices
        return all_prices



    def predict_next_price(self, historical_prices):
        """
        Applies linear regression to predict the next price based on historical prices.
        """
        if len(historical_prices) < 2:
            return None
        X = np.arange(len(historical_prices)).reshape(-1, 1)
        y = np.array(historical_prices).reshape(-1, 1)
        coefficients = np.polyfit(X.flatten(), y.flatten(), 1)
        model = np.poly1d(coefficients)
        predicted_price = model(len(historical_prices))
        return predicted_price


    def calculate_moving_average(self, prices, window=5):
        """Calculate the moving average for the given price list and window size."""
        if len(prices) < window:
            return prices[-1]  # Not enough data, return the last price.
        return np.mean(prices[-window:])


    def run(self, state: TradingState):
        # Only method required. It takes all buy and sell orders for all symbols as an input, and outputs a list of orders to be sent
        self.logger.print("traderData: " + state.traderData)
        self.logger.print("Observations: " + str(state.observations))

        result = {}

        for product in state.order_depths:
            if product == "AMETHYSTS":
                order_depth: OrderDepth = state.order_depths[product]
                orders: List[Order] = []

                buy_price = 9998  # Adjusted buy price
                sell_price = 10002  # Adjusted sell price

                self.logger.print("Buy price : " + str(buy_price))
                self.logger.print("Sell price : " + str(sell_price))
                self.logger.print("Buy Order depth : " + str(len(order_depth.buy_orders)) + ", Sell order depth : " + str(len(order_depth.sell_orders)))

                current_position = state.position.get(product, 0)
                self.logger.print("Current position: " + str(current_position))

                if current_position < self.POSITION_LIMIT[product]:
                    if len(order_depth.sell_orders) != 0:
                        best_ask, best_ask_amount = list(order_depth.sell_orders.items())[0]
                        if int(best_ask) <= buy_price:
                            quantity = min(-best_ask_amount, self.POSITION_LIMIT[product] - current_position)
                            self.logger.print("BUY", str(quantity) + "x", best_ask)
                            orders.append(Order(product, best_ask, quantity))

                if current_position > 0:
                    if len(order_depth.buy_orders) != 0:
                        best_bid, best_bid_amount = list(order_depth.buy_orders.items())[0]
                        if int(best_bid) >= sell_price:
                            quantity = min(best_bid_amount, current_position)
                            self.logger.print("SELL", str(quantity) + "x", best_bid)
                            orders.append(Order(product, best_bid, -quantity))

                result[product] = orders
            
            # Inside the Trader class's run method, after handling AMETHYSTS
            elif product == "STARFRUIT":
                self.logger.print("Processing STARFRUIT...")
                trade_quantity = 10  # Define a consistent trade quantity for STARFRUIT

                # Retrieve historical price data for STARFRUIT and log for debugging
                historical_prices_starfruit = self.read_price_data(Trader.price_files, 'STARFRUIT')
                if not historical_prices_starfruit:
                    self.logger.print("No historical prices found for STARFRUIT.")
                    continue  # Skip STARFRUIT trading logic if no historical data found

                # Calculate the latest and predicted prices for STARFRUIT
                latest_price_starfruit = historical_prices_starfruit[-1]
                predicted_price_starfruit = self.predict_next_price(historical_prices_starfruit)
                self.logger.print(f"Latest price for STARFRUIT: {latest_price_starfruit}, Predicted price: {predicted_price_starfruit}")

                # Proceed only if there's a valid predicted price for STARFRUIT
                if predicted_price_starfruit is not None and latest_price_starfruit is not None:
                    current_position = state.position.get(product, 0)
                    order_depth = state.order_depths.get(product, None)

                    if order_depth:
                        self.logger.print(f"Current position for STARFRUIT: {current_position}")
                        # Determine if buying or selling based on predicted vs. latest price
                        if predicted_price_starfruit > latest_price_starfruit and order_depth.sell_orders:
                            best_ask_price = min(order_depth.sell_orders.keys())
                            if current_position + trade_quantity <= self.POSITION_LIMIT[product]:
                                self.logger.print(f"Buying STARFRUIT: {trade_quantity} at {best_ask_price}")
                                orders.append(Order(product, best_ask_price, trade_quantity))
                        elif predicted_price_starfruit < latest_price_starfruit and order_depth.buy_orders:
                            best_bid_price = max(order_depth.buy_orders.keys())
                            if current_position - trade_quantity >= -self.POSITION_LIMIT[product]:
                                self.logger.print(f"Selling STARFRUIT: {trade_quantity} at {best_bid_price}")
                                orders.append(Order(product, best_bid_price, -trade_quantity))

                        # Update the result dictionary with the orders for STARFRUIT
                        result[product] = orders
                    else:
                        self.logger.print(f"Order depth information missing for STARFRUIT.")
                else:
                    self.logger.print("Insufficient data for STARFRUIT trading decisions.")

                self.logger.print(f"Completed STARFRUIT processing. Current orders: {orders}")


        traderData = "SAMPLE"  # Placeholder for trader state data
        conversions = 1

        self.logger.flush(state, result, conversions, traderData)

        return result, conversions, traderData
