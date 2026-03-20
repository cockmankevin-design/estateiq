#pragma once

#include "types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace estateiq {

// ─── Risk limits ──────────────────────────────────────────────────────────────
struct RiskLimits {
    double max_position_weight{0.20};      // max single position % of NAV
    double max_gross_exposure{1.50};       // max sum(|weights|)
    double max_net_exposure{1.00};         // max net long/short
    double max_daily_loss{0.05};           // max daily P&L drawdown (fraction of NAV)
    double max_drawdown{0.15};             // max historical drawdown
    double max_var_95{0.03};              // max 1-day 95% VaR (fraction of NAV)
    double max_order_notional{500'000.0};  // max single order dollar value
    std::size_t max_open_orders{100};
    std::vector<std::string> blacklist;    // symbols not allowed to trade
};

// ─── Risk check result ────────────────────────────────────────────────────────
struct RiskCheck {
    bool   passed{true};
    std::string violation;  // empty when passed
};

// ─── Portfolio risk snapshot ──────────────────────────────────────────────────
struct RiskSnapshot {
    double nav{0.0};
    double gross_exposure{0.0};
    double net_exposure{0.0};
    double daily_pnl{0.0};
    double daily_pnl_pct{0.0};
    double current_drawdown{0.0};
    double var_95{0.0};
    double largest_position_weight{0.0};
    std::size_t n_open_positions{0};
    bool   trading_halted{false};
    std::string halt_reason;
};

// ─── Risk Manager ─────────────────────────────────────────────────────────────
class RiskManager {
public:
    explicit RiskManager(RiskLimits limits = {});

    // Pre-trade check — call before submitting an order
    RiskCheck check_order(
        const Order& order,
        const std::unordered_map<std::string, Position>& positions,
        double nav,
        Price market_price
    ) const;

    // Pre-signal check — called when strategy emits a signal
    RiskCheck check_signal(
        const Signal& signal,
        const std::unordered_map<std::string, Position>& positions,
        double nav
    ) const;

    // Update risk state on each new bar / P&L event
    void update(double nav, double daily_pnl, double peak_nav);

    // Compute current risk snapshot
    RiskSnapshot snapshot(
        const std::unordered_map<std::string, Position>& positions,
        double nav
    ) const;

    // Emergency halt
    void halt(const std::string& reason);
    void resume();
    bool is_halted() const noexcept { return halted_; }

    // VaR estimation (historical simulation)
    double compute_var(
        const std::vector<double>& daily_returns,
        double confidence = 0.95
    ) const;

    const RiskLimits& limits() const noexcept { return limits_; }
    void set_limits(const RiskLimits& limits) { limits_ = limits; }

private:
    RiskLimits limits_;
    bool       halted_{false};
    std::string halt_reason_;
    double     daily_pnl_{0.0};
    double     current_drawdown_{0.0};

    bool is_blacklisted(const std::string& symbol) const;
    double position_weight(const Position& pos, double nav) const noexcept;
};

}  // namespace estateiq
