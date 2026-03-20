#pragma once

#include "types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace estateiq {

// ─── Portfolio snapshot ───────────────────────────────────────────────────────
struct PortfolioSnapshot {
    double cash{0.0};
    double invested_value{0.0};
    double nav{0.0};       // Net Asset Value = cash + invested
    double gross_exposure{0.0};
    double net_exposure{0.0};
    double total_realised_pnl{0.0};
    double total_unrealised_pnl{0.0};
    std::unordered_map<std::string, Position> positions;
};

// ─── Portfolio ────────────────────────────────────────────────────────────────
class Portfolio {
public:
    explicit Portfolio(double initial_cash = 100'000.0);

    // Apply a fill: updates positions, cash, realised P&L
    void apply_fill(const Fill& fill);

    // Update mark-to-market prices (call on each bar)
    void mark_to_market(const std::unordered_map<std::string, Price>& prices);

    // Query
    double       cash()     const noexcept { return cash_; }
    double       nav()      const noexcept;
    double       realised_pnl()   const noexcept { return total_realised_pnl_; }
    double       unrealised_pnl() const noexcept;

    const Position* position(const std::string& symbol) const;
    bool has_position(const std::string& symbol) const;

    PortfolioSnapshot snapshot() const;

    // Target weight rebalancing helper
    // Returns a set of Orders needed to move from current to target weights
    std::vector<Order> rebalance_orders(
        const std::unordered_map<std::string, Weight>& target_weights,
        const std::unordered_map<std::string, Price>&  prices,
        StrategyId strategy_id = 0
    ) const;

    // History
    void record_nav(TimePoint ts);
    const std::vector<std::pair<TimePoint, double>>& nav_history() const noexcept {
        return nav_history_;
    }

    double peak_nav() const noexcept { return peak_nav_; }

private:
    double cash_{0.0};
    double initial_cash_{0.0};
    double total_realised_pnl_{0.0};
    double peak_nav_{0.0};

    std::unordered_map<std::string, Position> positions_;
    std::vector<std::pair<TimePoint, double>> nav_history_;

    void update_position_on_fill(const Fill& fill);
    double compute_realised_pnl(const Position& pos, const Fill& fill) const;
};

}  // namespace estateiq
