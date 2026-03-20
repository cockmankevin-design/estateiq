#include <gtest/gtest.h>
#include "order_execution.hpp"

using namespace estateiq;

// Helper: build a Bar for a given symbol at close price
static Bar make_bar(const std::string& sym, double close) {
    Bar b;
    b.symbol    = sym;
    b.timestamp = Clock::now();
    b.open = b.high = b.low = b.close = close;
    b.volume = 1e6;
    return b;
}

// Helper: build a basic market buy order
static Order make_market_order(const std::string& sym, double qty, Side side) {
    Order o;
    o.symbol   = sym;
    o.type     = OrderType::MARKET;
    o.side     = side;
    o.quantity = qty;
    return o;
}

// ─── Submit & fill ────────────────────────────────────────────────────────────
TEST(OrderExecution, MarketOrderFillsOnNextBar) {
    OrderExecutionEngine engine;
    int fill_count = 0;
    engine.on_fill([&](const Fill&) { ++fill_count; });

    Order order = make_market_order("AAPL", 10.0, Side::BUY);
    OrderId id  = engine.submit(order);
    ASSERT_GT(id, 0u);
    EXPECT_EQ(fill_count, 0) << "Should not fill before bar arrives";

    engine.on_bar(make_bar("AAPL", 150.0));
    EXPECT_EQ(fill_count, 1) << "Market order should fill on bar";
}

TEST(OrderExecution, OrderIdIsUnique) {
    OrderExecutionEngine engine;
    OrderId id1 = engine.submit(make_market_order("AAPL", 10.0, Side::BUY));
    OrderId id2 = engine.submit(make_market_order("AAPL", 10.0, Side::BUY));
    EXPECT_NE(id1, id2);
    EXPECT_GT(id2, id1);
}

TEST(OrderExecution, ZeroQuantityOrderRejected) {
    OrderExecutionEngine engine;
    int reject_count = 0;
    engine.on_reject([&](const Order&, const std::string&) { ++reject_count; });

    Order order = make_market_order("AAPL", 0.0, Side::BUY);
    OrderId id  = engine.submit(order);
    EXPECT_EQ(id, 0u)       << "Zero quantity should be rejected";
    EXPECT_EQ(reject_count, 1);
}

TEST(OrderExecution, EmptySymbolOrderRejected) {
    OrderExecutionEngine engine;
    int reject_count = 0;
    engine.on_reject([&](const Order&, const std::string&) { ++reject_count; });

    Order order = make_market_order("", 10.0, Side::BUY);
    engine.submit(order);
    EXPECT_EQ(reject_count, 1);
}

TEST(OrderExecution, CancelOpenOrder) {
    OrderExecutionEngine engine;

    // Submit a limit order (won't fill unless bar price matches)
    Order order = make_market_order("TSLA", 5.0, Side::BUY);
    order.type        = OrderType::LIMIT;
    order.limit_price = 1.0;  // far below market — won't fill easily
    OrderId id = engine.submit(order);
    ASSERT_GT(id, 0u);

    bool cancelled = engine.cancel(id);
    EXPECT_TRUE(cancelled);

    const Order* found = engine.find_order(id);
    ASSERT_NE(found, nullptr);
    EXPECT_EQ(found->status, OrderStatus::CANCELLED);
}

TEST(OrderExecution, LimitOrderFillsWhenBelowLow) {
    OrderExecutionEngine engine;
    int fill_count = 0;
    engine.on_fill([&](const Fill&) { ++fill_count; });

    Order order;
    order.symbol      = "MSFT";
    order.type        = OrderType::LIMIT;
    order.side        = Side::BUY;
    order.quantity    = 5.0;
    order.limit_price = 200.0;
    engine.submit(order);

    // Bar with low < limit_price
    Bar bar;
    bar.symbol    = "MSFT";
    bar.timestamp = Clock::now();
    bar.open = 202.0; bar.high = 205.0; bar.low = 198.0; bar.close = 203.0;
    engine.on_bar(bar);
    EXPECT_EQ(fill_count, 1);
}

TEST(OrderExecution, BuyOrderNoFillWhenSymbolMismatch) {
    OrderExecutionEngine engine;
    int fill_count = 0;
    engine.on_fill([&](const Fill&) { ++fill_count; });

    engine.submit(make_market_order("AAPL", 10.0, Side::BUY));
    engine.on_bar(make_bar("MSFT", 300.0));  // different symbol
    EXPECT_EQ(fill_count, 0);
}

TEST(OrderExecution, TotalStats) {
    OrderExecutionEngine engine;
    engine.submit(make_market_order("AAPL", 10.0, Side::BUY));
    engine.submit(make_market_order("AAPL", 5.0,  Side::SELL));
    engine.on_bar(make_bar("AAPL", 150.0));

    EXPECT_EQ(engine.total_fills(), 2u);
    EXPECT_GT(engine.total_commission(), 0.0);
    EXPECT_EQ(engine.total_orders(), 3u);  // ids start at 1, incremented twice
}
