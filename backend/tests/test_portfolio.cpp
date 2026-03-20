#include <gtest/gtest.h>
#include "portfolio.hpp"

using namespace estateiq;

// Helper: build a fill
static Fill make_fill(const std::string& sym, Side side, double qty, double price) {
    Fill f;
    f.id          = 1;
    f.symbol      = sym;
    f.side        = side;
    f.quantity    = qty;
    f.price       = price;
    f.commission  = qty * price * 0.001;
    f.slippage    = 0.0;
    f.executed_at = Clock::now();
    return f;
}

// ─── Cash management ──────────────────────────────────────────────────────────
TEST(Portfolio, InitialCashCorrect) {
    Portfolio p(50'000.0);
    EXPECT_DOUBLE_EQ(p.cash(), 50'000.0);
    EXPECT_NEAR(p.nav(), 50'000.0, 1e-6);
}

TEST(Portfolio, BuyReducesCash) {
    Portfolio p(100'000.0);
    double initial_cash = p.cash();
    Fill f = make_fill("AAPL", Side::BUY, 10.0, 150.0);
    p.apply_fill(f);

    double cost = 10.0 * 150.0 + f.commission;
    EXPECT_NEAR(p.cash(), initial_cash - cost, 1e-4);
}

TEST(Portfolio, SellIncreasesCash) {
    Portfolio p(100'000.0);
    // First buy
    p.apply_fill(make_fill("AAPL", Side::BUY, 10.0, 150.0));
    double cash_after_buy = p.cash();
    // Then sell
    Fill sell = make_fill("AAPL", Side::SELL, 10.0, 160.0);
    p.apply_fill(sell);
    double expected = cash_after_buy + 10.0 * 160.0 - sell.commission;
    EXPECT_NEAR(p.cash(), expected, 1e-4);
}

// ─── Position management ──────────────────────────────────────────────────────
TEST(Portfolio, BuyCreatesPosition) {
    Portfolio p(100'000.0);
    p.apply_fill(make_fill("MSFT", Side::BUY, 5.0, 300.0));
    EXPECT_TRUE(p.has_position("MSFT"));
    const Position* pos = p.position("MSFT");
    ASSERT_NE(pos, nullptr);
    EXPECT_DOUBLE_EQ(pos->quantity, 5.0);
    EXPECT_NEAR(pos->avg_cost, 300.0, 1e-6);
}

TEST(Portfolio, FullSellFlattensPosition) {
    Portfolio p(100'000.0);
    p.apply_fill(make_fill("GOOG", Side::BUY,  10.0, 2800.0));
    p.apply_fill(make_fill("GOOG", Side::SELL, 10.0, 2900.0));
    EXPECT_FALSE(p.has_position("GOOG"));
}

TEST(Portfolio, PartialSellReducesQuantity) {
    Portfolio p(100'000.0);
    p.apply_fill(make_fill("AMZN", Side::BUY,  10.0, 3500.0));
    p.apply_fill(make_fill("AMZN", Side::SELL,  4.0, 3600.0));
    const Position* pos = p.position("AMZN");
    ASSERT_NE(pos, nullptr);
    EXPECT_NEAR(pos->quantity, 6.0, 1e-6);
}

// ─── NAV & P&L ────────────────────────────────────────────────────────────────
TEST(Portfolio, NAVUpdatesOnMarkToMarket) {
    Portfolio p(100'000.0);
    p.apply_fill(make_fill("SPY", Side::BUY, 100.0, 400.0));
    // Mark up 10%
    p.mark_to_market({{"SPY", 440.0}});
    // Unrealised P&L = 100 * (440 - 400) = 4000
    EXPECT_NEAR(p.unrealised_pnl(), 4'000.0, 1e-4);
}

TEST(Portfolio, RealisedPnLOnSell) {
    Portfolio p(100'000.0);
    p.apply_fill(make_fill("NVDA", Side::BUY,  10.0, 500.0));
    p.apply_fill(make_fill("NVDA", Side::SELL, 10.0, 600.0));
    // Realised = 10 * (600 - 500) = 1000
    EXPECT_NEAR(p.realised_pnl(), 1'000.0, 1e-2);
}

TEST(Portfolio, PeakNAVTracked) {
    Portfolio p(100'000.0);
    // NAV starts at 100k
    p.mark_to_market({});
    EXPECT_GE(p.peak_nav(), 100'000.0);

    // Buy and mark up — NAV rises
    p.apply_fill(make_fill("SPY", Side::BUY, 10.0, 400.0));
    p.mark_to_market({{"SPY", 500.0}});
    double peak = p.peak_nav();
    EXPECT_GT(peak, 100'000.0);

    // Mark down — peak unchanged
    p.mark_to_market({{"SPY", 350.0}});
    EXPECT_NEAR(p.peak_nav(), peak, 1e-4);
}

// ─── Rebalance orders ─────────────────────────────────────────────────────────
TEST(Portfolio, RebalanceOrdersBuyWhenUnderweight) {
    Portfolio p(100'000.0);
    // No positions currently
    std::unordered_map<std::string, Weight> targets = {{"AAPL", 0.50}};
    std::unordered_map<std::string, Price>  prices  = {{"AAPL", 150.0}};

    auto orders = p.rebalance_orders(targets, prices);
    ASSERT_FALSE(orders.empty());
    EXPECT_EQ(orders[0].side, Side::BUY);
    EXPECT_GT(orders[0].quantity, 0.0);
}

TEST(Portfolio, RebalanceOrdersSellWhenOverweight) {
    Portfolio p(100'000.0);
    // Buy 100 shares at $500 = $50k (50% of NAV)
    p.apply_fill(make_fill("SPY", Side::BUY, 100.0, 500.0));
    p.mark_to_market({{"SPY", 500.0}});

    // Target 20% — need to sell
    std::unordered_map<std::string, Weight> targets = {{"SPY", 0.20}};
    std::unordered_map<std::string, Price>  prices  = {{"SPY", 500.0}};

    auto orders = p.rebalance_orders(targets, prices);
    ASSERT_FALSE(orders.empty());
    EXPECT_EQ(orders[0].side, Side::SELL);
}

TEST(Portfolio, NoRebalanceWhenAlreadyAtTarget) {
    Portfolio p(100'000.0);
    // Buy exactly 10% worth at $100 = $10k out of $100k NAV
    p.apply_fill(make_fill("XYZ", Side::BUY, 100.0, 100.0));
    p.mark_to_market({{"XYZ", 100.0}});

    std::unordered_map<std::string, Weight> targets = {{"XYZ", 0.10}};
    std::unordered_map<std::string, Price>  prices  = {{"XYZ", 100.0}};

    // Difference is tiny — no order expected
    auto orders = p.rebalance_orders(targets, prices);
    // Orders under $10 threshold should not generate an order
    for (auto& ord : orders) {
        EXPECT_LT(ord.quantity * 100.0, 10.0) << "Tiny rebalance should be ignored";
    }
}
