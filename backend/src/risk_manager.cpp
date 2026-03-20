#include "risk_manager.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace estateiq {

RiskManager::RiskManager(RiskLimits limits)
    : limits_(std::move(limits)) {}

// ─── Pre-trade order check ────────────────────────────────────────────────────
RiskCheck RiskManager::check_order(
    const Order& order,
    const std::unordered_map<std::string, Position>& positions,
    double nav,
    Price market_price
) const {
    if (halted_) return {false, "Trading halted: " + halt_reason_};

    if (is_blacklisted(order.symbol))
        return {false, "Symbol blacklisted: " + order.symbol};

    double notional = order.quantity * market_price;
    if (notional > limits_.max_order_notional)
        return {false, "Order notional exceeds limit"};

    // Check resulting position weight
    double new_qty = order.quantity * (order.side == Side::BUY ? 1.0 : -1.0);
    auto it = positions.find(order.symbol);
    double current_qty = it != positions.end() ? it->second.quantity : 0.0;
    double new_position_value = std::abs(current_qty + new_qty) * market_price;
    if (nav > 0 && new_position_value / nav > limits_.max_position_weight)
        return {false, "Position weight would exceed limit"};

    return {true, ""};
}

// ─── Pre-signal check ─────────────────────────────────────────────────────────
RiskCheck RiskManager::check_signal(
    const Signal& signal,
    const std::unordered_map<std::string, Position>& positions,
    double nav
) const {
    if (halted_) return {false, "Trading halted: " + halt_reason_};
    if (is_blacklisted(signal.symbol))
        return {false, "Symbol blacklisted: " + signal.symbol};
    if (signal.target_weight > limits_.max_position_weight)
        return {false, "Target weight exceeds position limit"};
    return {true, ""};
}

// ─── State update ─────────────────────────────────────────────────────────────
void RiskManager::update(double nav, double daily_pnl, double peak_nav) {
    daily_pnl_ = daily_pnl;
    if (peak_nav > 0)
        current_drawdown_ = (nav - peak_nav) / peak_nav;  // negative when underwater

    // Auto-halt on daily loss limit
    if (nav > 0 && daily_pnl / nav < -limits_.max_daily_loss && !halted_) {
        halt("Daily loss limit breached");
    }
    // Auto-halt on max drawdown
    if (current_drawdown_ < -limits_.max_drawdown && !halted_) {
        halt("Maximum drawdown breached");
    }
}

// ─── Snapshot ─────────────────────────────────────────────────────────────────
RiskSnapshot RiskManager::snapshot(
    const std::unordered_map<std::string, Position>& positions,
    double nav
) const {
    RiskSnapshot snap;
    snap.nav              = nav;
    snap.daily_pnl        = daily_pnl_;
    snap.daily_pnl_pct    = nav > 0 ? daily_pnl_ / nav : 0.0;
    snap.current_drawdown = current_drawdown_;
    snap.trading_halted   = halted_;
    snap.halt_reason      = halt_reason_;
    snap.n_open_positions = 0;

    double gross = 0.0, net = 0.0, largest_w = 0.0;
    for (auto& [sym, pos] : positions) {
        if (!pos.is_flat()) {
            ++snap.n_open_positions;
            double mv  = std::abs(pos.market_value());
            double w   = nav > 0 ? mv / nav : 0.0;
            gross += mv;
            net   += pos.market_value();
            largest_w = std::max(largest_w, w);
        }
    }
    snap.gross_exposure           = nav > 0 ? gross / nav : 0.0;
    snap.net_exposure             = nav > 0 ? net   / nav : 0.0;
    snap.largest_position_weight  = largest_w;
    return snap;
}

// ─── Halt / resume ────────────────────────────────────────────────────────────
void RiskManager::halt(const std::string& reason) {
    halted_      = true;
    halt_reason_ = reason;
    spdlog::warn("[RiskManager] TRADING HALTED: {}", reason);
}
void RiskManager::resume() {
    halted_      = false;
    halt_reason_ = "";
    spdlog::info("[RiskManager] Trading resumed");
}

// ─── VaR ──────────────────────────────────────────────────────────────────────
double RiskManager::compute_var(
    const std::vector<double>& daily_returns,
    double confidence
) const {
    if (daily_returns.empty()) return 0.0;
    std::vector<double> sorted = daily_returns;
    std::sort(sorted.begin(), sorted.end());
    std::size_t idx = static_cast<std::size_t>((1.0 - confidence) * sorted.size());
    return sorted[std::min(idx, sorted.size() - 1)];
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool RiskManager::is_blacklisted(const std::string& symbol) const {
    return std::find(limits_.blacklist.begin(), limits_.blacklist.end(), symbol)
           != limits_.blacklist.end();
}

double RiskManager::position_weight(const Position& pos, double nav) const noexcept {
    return nav > 0 ? std::abs(pos.market_value()) / nav : 0.0;
}

}  // namespace estateiq
