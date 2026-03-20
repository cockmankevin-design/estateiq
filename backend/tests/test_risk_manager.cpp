#include <gtest/gtest.h>
#include "risk_manager.hpp"

using namespace estateiq;

// Helper: build a simple long signal
static Signal make_signal(const std::string& sym, double weight) {
    Signal s;
    s.symbol        = sym;
    s.type          = SignalType::LONG;
    s.target_weight = weight;
    s.timestamp     = Clock::now();
    return s;
}

// Helper: build a simple market buy order
static Order make_order(const std::string& sym, double qty, double price) {
    Order o;
    o.symbol   = sym;
    o.type     = OrderType::MARKET;
    o.side     = Side::BUY;
    o.quantity = qty;
    o.limit_price = price;
    return o;
}

// ─── Basic checks ─────────────────────────────────────────────────────────────
TEST(RiskManager, SignalPassesByDefault) {
    RiskManager rm;
    std::unordered_map<std::string, Position> positions;
    auto check = rm.check_signal(make_signal("AAPL", 0.10), positions, 100'000.0);
    EXPECT_TRUE(check.passed);
}

TEST(RiskManager, SignalFailsOnBlacklist) {
    RiskLimits limits;
    limits.blacklist = {"FORBIDDEN"};
    RiskManager rm(limits);
    std::unordered_map<std::string, Position> positions;
    auto check = rm.check_signal(make_signal("FORBIDDEN", 0.05), positions, 100'000.0);
    EXPECT_FALSE(check.passed);
    EXPECT_FALSE(check.violation.empty());
}

TEST(RiskManager, SignalFailsOnWeightExcess) {
    RiskLimits limits;
    limits.max_position_weight = 0.10;
    RiskManager rm(limits);
    std::unordered_map<std::string, Position> positions;
    auto check = rm.check_signal(make_signal("AAPL", 0.50), positions, 100'000.0);
    EXPECT_FALSE(check.passed);
}

TEST(RiskManager, OrderPassesNotionalCheck) {
    RiskLimits limits;
    limits.max_order_notional = 1'000'000.0;
    RiskManager rm(limits);
    std::unordered_map<std::string, Position> positions;
    // 100 shares @ $500 = $50k — under limit
    auto check = rm.check_order(make_order("TSLA", 100.0, 500.0), positions, 200'000.0, 500.0);
    EXPECT_TRUE(check.passed);
}

TEST(RiskManager, OrderFailsNotionalCheck) {
    RiskLimits limits;
    limits.max_order_notional = 1'000.0;  // very low limit
    RiskManager rm(limits);
    std::unordered_map<std::string, Position> positions;
    // 100 shares @ $500 = $50k — exceeds $1k limit
    auto check = rm.check_order(make_order("TSLA", 100.0, 500.0), positions, 200'000.0, 500.0);
    EXPECT_FALSE(check.passed);
}

// ─── Halt / resume ────────────────────────────────────────────────────────────
TEST(RiskManager, HaltBlocksSignals) {
    RiskManager rm;
    rm.halt("Test halt");
    EXPECT_TRUE(rm.is_halted());

    std::unordered_map<std::string, Position> positions;
    auto check = rm.check_signal(make_signal("AAPL", 0.10), positions, 100'000.0);
    EXPECT_FALSE(check.passed);
    EXPECT_NE(check.violation.find("halted"), std::string::npos);
}

TEST(RiskManager, ResumeAllowsSignals) {
    RiskManager rm;
    rm.halt("Test");
    rm.resume();
    EXPECT_FALSE(rm.is_halted());

    std::unordered_map<std::string, Position> positions;
    auto check = rm.check_signal(make_signal("AAPL", 0.10), positions, 100'000.0);
    EXPECT_TRUE(check.passed);
}

TEST(RiskManager, AutoHaltOnDailyLoss) {
    RiskLimits limits;
    limits.max_daily_loss = 0.02;  // 2%
    RiskManager rm(limits);

    double nav  = 100'000.0;
    double peak = nav;
    // Daily loss of 3% — exceeds 2% limit
    rm.update(nav, -3'000.0, peak);
    EXPECT_TRUE(rm.is_halted());
}

TEST(RiskManager, AutoHaltOnMaxDrawdown) {
    RiskLimits limits;
    limits.max_drawdown  = 0.10;  // 10%
    limits.max_daily_loss = 1.0;  // disable daily loss limit for this test
    RiskManager rm(limits);

    double peak = 100'000.0;
    double nav  = 88'000.0;   // 12% underwater
    rm.update(nav, 0.0, peak);
    EXPECT_TRUE(rm.is_halted());
}

// ─── VaR ──────────────────────────────────────────────────────────────────────
TEST(RiskManager, VaRComputation) {
    RiskManager rm;
    std::vector<double> returns(1000);
    for (int i = 0; i < 1000; ++i)
        returns[i] = -0.01 + (i % 20) * 0.001;
    double v = rm.compute_var(returns, 0.95);
    EXPECT_LT(v, 0.0) << "VaR should be negative (loss)";
}

// ─── Snapshot ─────────────────────────────────────────────────────────────────
TEST(RiskManager, SnapshotNAVReflectsInput) {
    RiskManager rm;
    std::unordered_map<std::string, Position> positions;
    double nav = 123'456.0;
    auto snap = rm.snapshot(positions, nav);
    EXPECT_DOUBLE_EQ(snap.nav, nav);
    EXPECT_EQ(snap.n_open_positions, 0u);
    EXPECT_FALSE(snap.trading_halted);
}
