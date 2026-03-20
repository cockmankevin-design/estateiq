#include "portfolio.hpp"
#include <spdlog/spdlog.h>
#include <cmath>
#include <stdexcept>

namespace estateiq {

Portfolio::Portfolio(double initial_cash)
    : cash_(initial_cash)
    , initial_cash_(initial_cash)
    , peak_nav_(initial_cash)
{}

// ─── Apply fill ───────────────────────────────────────────────────────────────
void Portfolio::apply_fill(const Fill& fill) {
    update_position_on_fill(fill);

    double cost = fill.quantity * fill.price;
    double total_cost = cost + fill.commission + fill.slippage;

    if (fill.side == Side::BUY) {
        cash_ -= total_cost;
    } else {
        cash_ += cost - fill.commission - fill.slippage;
    }

    spdlog::debug("Portfolio cash after fill: {:.2f}", cash_);
}

// ─── Mark-to-market ───────────────────────────────────────────────────────────
void Portfolio::mark_to_market(const std::unordered_map<std::string, Price>& prices) {
    for (auto& [sym, pos] : positions_) {
        auto it = prices.find(sym);
        if (it == prices.end()) continue;
        pos.market_price    = it->second;
        pos.unrealised_pnl  = (it->second - pos.avg_cost) * pos.quantity;
    }
    double current_nav = nav();
    peak_nav_ = std::max(peak_nav_, current_nav);
}

// ─── NAV ──────────────────────────────────────────────────────────────────────
double Portfolio::nav() const noexcept {
    double invested = 0.0;
    for (auto& [sym, pos] : positions_)
        invested += pos.market_value();
    return cash_ + invested;
}

double Portfolio::unrealised_pnl() const noexcept {
    double total = 0.0;
    for (auto& [sym, pos] : positions_)
        total += pos.unrealised_pnl;
    return total;
}

// ─── Query ────────────────────────────────────────────────────────────────────
const Position* Portfolio::position(const std::string& symbol) const {
    auto it = positions_.find(symbol);
    return it != positions_.end() ? &it->second : nullptr;
}

bool Portfolio::has_position(const std::string& symbol) const {
    auto it = positions_.find(symbol);
    return it != positions_.end() && !it->second.is_flat();
}

PortfolioSnapshot Portfolio::snapshot() const {
    PortfolioSnapshot s;
    s.cash      = cash_;
    s.positions = positions_;
    s.nav       = nav();
    s.total_realised_pnl   = total_realised_pnl_;
    s.total_unrealised_pnl = unrealised_pnl();

    double gross = 0.0, net = 0.0;
    for (auto& [sym, pos] : positions_) {
        gross += std::abs(pos.market_value());
        net   += pos.market_value();
    }
    s.invested_value = gross;
    s.gross_exposure = s.nav > 0 ? gross / s.nav : 0.0;
    s.net_exposure   = s.nav > 0 ? net   / s.nav : 0.0;
    return s;
}

void Portfolio::record_nav(TimePoint ts) {
    nav_history_.emplace_back(ts, nav());
}

// ─── Rebalance orders ─────────────────────────────────────────────────────────
std::vector<Order> Portfolio::rebalance_orders(
    const std::unordered_map<std::string, Weight>& target_weights,
    const std::unordered_map<std::string, Price>& prices,
    StrategyId strategy_id
) const {
    double current_nav = nav();
    std::vector<Order> orders;

    for (auto& [sym, target_w] : target_weights) {
        auto price_it = prices.find(sym);
        if (price_it == prices.end() || price_it->second <= 0) continue;

        double target_value = current_nav * target_w;
        auto pos_it         = positions_.find(sym);
        double current_qty  = pos_it != positions_.end() ? pos_it->second.quantity : 0.0;
        double current_val  = current_qty * price_it->second;
        double delta_val    = target_value - current_val;

        if (std::abs(delta_val) < 10.0) continue;  // ignore tiny rebalances

        Order order;
        order.strategy_id = strategy_id;
        order.symbol      = sym;
        order.type        = OrderType::MARKET;
        order.quantity    = std::abs(delta_val) / price_it->second;
        order.side        = delta_val > 0 ? Side::BUY : Side::SELL;
        orders.push_back(order);
    }
    return orders;
}

// ─── Internal helpers ─────────────────────────────────────────────────────────
void Portfolio::update_position_on_fill(const Fill& fill) {
    auto& pos = positions_[fill.symbol];
    if (pos.symbol.empty()) pos.symbol = fill.symbol;

    double qty_delta = fill.quantity * (fill.side == Side::BUY ? 1.0 : -1.0);

    // Update average cost (only on increases)
    if ((fill.side == Side::BUY && pos.quantity >= 0) ||
        (fill.side == Side::SELL && pos.quantity <= 0)) {
        double prev_cost  = pos.avg_cost * std::abs(pos.quantity);
        double fill_cost  = fill.price   * fill.quantity;
        double new_qty    = std::abs(pos.quantity) + fill.quantity;
        pos.avg_cost = new_qty > 0 ? (prev_cost + fill_cost) / new_qty : fill.price;
    }

    double realised = compute_realised_pnl(pos, fill);
    total_realised_pnl_ += realised;
    pos.realised_pnl    += realised;

    pos.quantity += qty_delta;
    pos.market_price = fill.price;

    // Remove flat positions
    if (std::abs(pos.quantity) < 1e-9) {
        positions_.erase(fill.symbol);
    }
}

double Portfolio::compute_realised_pnl(const Position& pos, const Fill& fill) const {
    // Only realised when reducing / reversing a position
    if (fill.side == Side::BUY && pos.quantity < 0) {
        double covered = std::min(fill.quantity, -pos.quantity);
        return covered * (pos.avg_cost - fill.price);
    }
    if (fill.side == Side::SELL && pos.quantity > 0) {
        double sold = std::min(fill.quantity, pos.quantity);
        return sold * (fill.price - pos.avg_cost);
    }
    return 0.0;
}

}  // namespace estateiq
