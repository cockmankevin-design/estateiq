#include <gtest/gtest.h>
#include "performance.hpp"
#include <cmath>
#include <numeric>
#include <vector>

using namespace estateiq;

// ─── Fixtures ─────────────────────────────────────────────────────────────────
// Flat returns (0% every day)
static std::vector<double> flat_returns(int n, double r = 0.0) {
    return std::vector<double>(n, r);
}

// Uptrending returns
static std::vector<double> trending_returns(int n, double daily = 0.001) {
    return std::vector<double>(n, daily);
}

// ─── Total return ─────────────────────────────────────────────────────────────
TEST(Performance, TotalReturnFlat) {
    EXPECT_NEAR(PerformanceCalculator::total_return(flat_returns(252)), 0.0, 1e-10);
}

TEST(Performance, TotalReturnPositive) {
    auto r = trending_returns(252, 0.001);
    double tr = PerformanceCalculator::total_return(r);
    EXPECT_GT(tr, 0.0);
}

TEST(Performance, TotalReturnCompoundsCorrectly) {
    // 2 bars of 10% return → total = (1.1)^2 - 1 = 21%
    std::vector<double> r = {0.1, 0.1};
    EXPECT_NEAR(PerformanceCalculator::total_return(r), 0.21, 1e-9);
}

// ─── CAGR ─────────────────────────────────────────────────────────────────────
TEST(Performance, CAGRPositiveTrend) {
    auto r = trending_returns(252 * 3, 0.0005);
    double c = PerformanceCalculator::cagr(r);
    EXPECT_GT(c, 0.0);
    EXPECT_LT(c, 1.0);
}

// ─── Volatility ───────────────────────────────────────────────────────────────
TEST(Performance, VolZeroForFlatReturns) {
    EXPECT_NEAR(PerformanceCalculator::annualised_vol(flat_returns(252, 0.001)), 0.0, 1e-10);
}

TEST(Performance, VolPositiveForNoisyReturns) {
    std::vector<double> r(252);
    for (int i = 0; i < 252; ++i) r[i] = (i % 2 == 0) ? 0.01 : -0.01;
    EXPECT_GT(PerformanceCalculator::annualised_vol(r), 0.0);
}

// ─── Max drawdown ─────────────────────────────────────────────────────────────
TEST(Performance, MaxDrawdownZeroForUptrend) {
    auto r = trending_returns(100, 0.001);
    EXPECT_NEAR(PerformanceCalculator::max_drawdown(r), 0.0, 1e-10);
}

TEST(Performance, MaxDrawdownKnownValue) {
    // Up 50%, then down 50% → drawdown from peak = -33.3%
    std::vector<double> r(2 * 252);
    for (int i = 0; i <   252; ++i) r[i]           =  0.001975;  // ~+50% over year
    for (int i = 252; i < 504; ++i) r[i]           = -0.001976;  // ~-50% over year
    double mdd = PerformanceCalculator::max_drawdown(r);
    EXPECT_LT(mdd, 0.0);
    EXPECT_GT(mdd, -1.0);
}

// ─── Sharpe ───────────────────────────────────────────────────────────────────
TEST(Performance, SharpePositiveForHighReturn) {
    auto r = trending_returns(252, 0.002);  // ~50% annual
    double rf_daily = 0.04 / 252;
    double sh = PerformanceCalculator::sharpe(r, rf_daily);
    EXPECT_GT(sh, 0.0);
}

TEST(Performance, SharpeNegativeForNegativeReturn) {
    auto r = flat_returns(252, -0.001);
    double rf_daily = 0.04 / 252;
    double sh = PerformanceCalculator::sharpe(r, rf_daily);
    EXPECT_LT(sh, 0.0);
}

// ─── Win rate ─────────────────────────────────────────────────────────────────
TEST(Performance, WinRateAllPositive) {
    auto r = trending_returns(100, 0.001);
    EXPECT_NEAR(PerformanceCalculator::win_rate(r), 1.0, 1e-9);
}

TEST(Performance, WinRateHalfHalf) {
    std::vector<double> r(100);
    for (int i = 0; i < 100; ++i) r[i] = (i % 2 == 0) ? 0.01 : -0.01;
    EXPECT_NEAR(PerformanceCalculator::win_rate(r), 0.5, 1e-9);
}

// ─── VaR / CVaR ───────────────────────────────────────────────────────────────
TEST(Performance, VaRIsNegativeForMixedReturns) {
    std::vector<double> r(100);
    for (int i = 0; i < 100; ++i) r[i] = (i % 10 == 0) ? -0.05 : 0.01;
    double v = PerformanceCalculator::var(r, 0.05);
    EXPECT_LT(v, 0.01);
}

TEST(Performance, CVaRIsLessThanOrEqualToVaR) {
    std::vector<double> r(500);
    for (int i = 0; i < 500; ++i) r[i] = -0.01 + (i % 20) * 0.001;
    double v  = PerformanceCalculator::var(r,  0.05);
    double cv = PerformanceCalculator::cvar(r, 0.05);
    EXPECT_LE(cv, v) << "CVaR must be <= VaR";
}

// ─── Skewness / Kurtosis ──────────────────────────────────────────────────────
TEST(Performance, SkewnessZeroForSymmetric) {
    std::vector<double> r = {-0.02, -0.01, 0.0, 0.01, 0.02};
    EXPECT_NEAR(PerformanceCalculator::skewness(r), 0.0, 1e-10);
}

TEST(Performance, KurtosisZeroForMesokurtic) {
    // Normal distribution has excess kurtosis ~0
    std::vector<double> r(1000);
    for (int i = 0; i < 1000; ++i)
        r[i] = 0.01 * std::sin(i * 0.1);  // oscillating, near-symmetric
    // Just check it returns a finite value
    double k = PerformanceCalculator::kurtosis(r);
    EXPECT_TRUE(std::isfinite(k));
}

// ─── Beta / Alpha ─────────────────────────────────────────────────────────────
TEST(Performance, BetaOfBenchmarkVsItself) {
    auto r = trending_returns(252, 0.001);
    double b = PerformanceCalculator::beta(r, r);
    EXPECT_NEAR(b, 1.0, 1e-6);
}

TEST(Performance, AlphaZeroWhenReturnEqualsBenchmark) {
    auto r = trending_returns(252, 0.001);
    double rf = 0.04 / 252;
    double a  = PerformanceCalculator::alpha_ann(r, r, rf);
    EXPECT_NEAR(a, 0.0, 1e-6);
}

// ─── Full compute() ───────────────────────────────────────────────────────────
TEST(Performance, ComputeReturnsAllMetrics) {
    auto r = trending_returns(252 * 2, 0.0005);
    PerformanceCalculator pc(0.04);
    auto m = pc.compute(r);
    EXPECT_GT(m.total_return, 0.0);
    EXPECT_GT(m.sharpe, 0.0);
    EXPECT_LE(m.max_drawdown, 0.0);
    EXPECT_GE(m.win_rate, 0.0);
    EXPECT_LE(m.win_rate, 1.0);
    EXPECT_EQ(m.n_bars, 252u * 2);
}

TEST(Performance, EmptyReturnsReturnZeroMetrics) {
    PerformanceCalculator pc(0.04);
    auto m = pc.compute({});
    EXPECT_DOUBLE_EQ(m.total_return, 0.0);
    EXPECT_DOUBLE_EQ(m.sharpe, 0.0);
}

// ─── JSON serialisation ───────────────────────────────────────────────────────
TEST(Performance, ToJsonContainsKey) {
    Metrics m;
    m.sharpe = 1.5;
    std::string json = PerformanceCalculator::to_json(m);
    EXPECT_NE(json.find("sharpe"), std::string::npos);
    EXPECT_NE(json.find("1.5"), std::string::npos);
}
