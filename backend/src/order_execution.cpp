#include "order_execution.hpp"
#include <spdlog/spdlog.h>
#include <algorithm>
#include <cmath>
#include <stdexcept>

namespace estateiq {

OrderExecutionEngine::OrderExecutionEngine(ExecutionConfig cfg)
    : cfg_(std::move(cfg)) {}

// ─── Submit ───────────────────────────────────────────────────────────────────
OrderId OrderExecutionEngine::submit(Order order) {
    std::string reason;
    if (!validate(order, reason)) {
        order.status = OrderStatus::REJECTED;
        spdlog::warn("Order rejected [{}]: {}", order.symbol, reason);
        emit_reject(order, reason);
        return 0;
    }

    order.id         = order_counter_.fetch_add(1, std::memory_order_relaxed);
    order.created_at = Clock::now();
    order.updated_at = order.created_at;
    order.status     = OrderStatus::OPEN;

    {
        std::lock_guard<std::mutex> lk(orders_mtx_);
        orders_[order.id] = order;
        symbol_orders_[order.symbol].push_back(order.id);
    }

    spdlog::info("Order #{} submitted: {} {} {} qty={:.2f}",
                 order.id, order.symbol,
                 order.side == Side::BUY ? "BUY" : "SELL",
                 order.type == OrderType::MARKET ? "MARKET" : "LIMIT",
                 order.quantity);

    // Market orders fill immediately in simulation
    if (cfg_.simulate && order.type == OrderType::MARKET) {
        spdlog::debug("Market order #{} queued for fill on next bar", order.id);
    }

    return order.id;
}

// ─── Cancel ───────────────────────────────────────────────────────────────────
bool OrderExecutionEngine::cancel(OrderId id) {
    std::lock_guard<std::mutex> lk(orders_mtx_);
    auto it = orders_.find(id);
    if (it == orders_.end()) return false;
    if (it->second.status != OrderStatus::OPEN &&
        it->second.status != OrderStatus::PENDING) {
        return false;
    }
    it->second.status     = OrderStatus::CANCELLED;
    it->second.updated_at = Clock::now();
    spdlog::info("Order #{} cancelled", id);
    return true;
}

// ─── On bar (fill simulation) ─────────────────────────────────────────────────
void OrderExecutionEngine::on_bar(const Bar& bar) {
    std::vector<OrderId> to_fill;
    {
        std::lock_guard<std::mutex> lk(orders_mtx_);
        auto sym_it = symbol_orders_.find(bar.symbol);
        if (sym_it == symbol_orders_.end()) return;
        for (OrderId oid : sym_it->second) {
            auto& ord = orders_.at(oid);
            if (ord.status != OrderStatus::OPEN) continue;

            bool should_fill = false;
            if (ord.type == OrderType::MARKET) {
                should_fill = true;
            } else if (ord.type == OrderType::LIMIT) {
                should_fill = (ord.side == Side::BUY  && bar.low  <= ord.limit_price) ||
                              (ord.side == Side::SELL && bar.high >= ord.limit_price);
            } else if (ord.type == OrderType::STOP) {
                should_fill = (ord.side == Side::BUY  && bar.high >= ord.stop_price) ||
                              (ord.side == Side::SELL && bar.low  <= ord.stop_price);
            }
            if (should_fill) to_fill.push_back(oid);
        }
    }

    for (OrderId oid : to_fill) {
        Order order_copy;
        {
            std::lock_guard<std::mutex> lk(orders_mtx_);
            order_copy = orders_.at(oid);
        }
        Fill fill = simulate_fill(order_copy, bar);
        {
            std::lock_guard<std::mutex> lk(fills_mtx_);
            fills_.push_back(fill);
        }
        update_order_status(oid, OrderStatus::FILLED);
        emit_fill(fill);
        spdlog::info("Fill #{}: {} {} {} @ {:.4f}",
                     fill.id, fill.symbol,
                     fill.side == Side::BUY ? "BUY" : "SELL",
                     fill.quantity, fill.price);
    }
}

// ─── Helpers ──────────────────────────────────────────────────────────────────
bool OrderExecutionEngine::validate(const Order& order, std::string& reason) const {
    if (order.symbol.empty()) { reason = "Empty symbol";       return false; }
    if (order.quantity <= 0)  { reason = "Quantity must be >0"; return false; }
    if (order.type == OrderType::LIMIT && order.limit_price <= 0) {
        reason = "Limit price must be >0"; return false;
    }
    if (order.quantity * order.limit_price > cfg_.max_order_value &&
        order.type != OrderType::MARKET) {
        reason = "Order exceeds max notional"; return false;
    }
    return true;
}

Fill OrderExecutionEngine::simulate_fill(const Order& order, const Bar& bar) {
    // Execution price: open of the bar + slippage
    Price exec_price = bar.open;
    double slip = exec_price * cfg_.slippage_bps / 10'000.0;
    exec_price += (order.side == Side::BUY ? slip : -slip);

    double notional   = order.quantity * exec_price;
    double commission = notional * cfg_.commission_rate;

    Fill fill;
    fill.id          = fill_counter_.fetch_add(1, std::memory_order_relaxed);
    fill.order_id    = order.id;
    fill.symbol      = order.symbol;
    fill.side        = order.side;
    fill.quantity    = order.quantity;
    fill.price       = exec_price;
    fill.commission  = commission;
    fill.slippage    = slip;
    fill.executed_at = bar.timestamp;
    return fill;
}

void OrderExecutionEngine::emit_fill(const Fill& fill) {
    for (auto& cb : fill_callbacks_) cb(fill);
}
void OrderExecutionEngine::emit_reject(const Order& order, const std::string& reason) {
    for (auto& cb : reject_callbacks_) cb(order, reason);
}
void OrderExecutionEngine::update_order_status(OrderId id, OrderStatus status) {
    std::lock_guard<std::mutex> lk(orders_mtx_);
    auto it = orders_.find(id);
    if (it != orders_.end()) {
        it->second.status     = status;
        it->second.updated_at = Clock::now();
    }
}

const Order* OrderExecutionEngine::find_order(OrderId id) const {
    std::lock_guard<std::mutex> lk(orders_mtx_);
    auto it = orders_.find(id);
    return it != orders_.end() ? &it->second : nullptr;
}

std::vector<Order> OrderExecutionEngine::open_orders() const {
    std::lock_guard<std::mutex> lk(orders_mtx_);
    std::vector<Order> out;
    for (auto& [id, ord] : orders_)
        if (ord.status == OrderStatus::OPEN) out.push_back(ord);
    return out;
}

std::vector<Fill> OrderExecutionEngine::fill_history() const {
    std::lock_guard<std::mutex> lk(fills_mtx_);
    return fills_;
}

std::size_t OrderExecutionEngine::total_fills() const noexcept {
    std::lock_guard<std::mutex> lk(fills_mtx_);
    return fills_.size();
}

double OrderExecutionEngine::total_commission() const noexcept {
    std::lock_guard<std::mutex> lk(fills_mtx_);
    double total = 0.0;
    for (auto& f : fills_) total += f.commission;
    return total;
}

}  // namespace estateiq
