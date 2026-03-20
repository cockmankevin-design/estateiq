#pragma once

#include "types.hpp"

#include <atomic>
#include <functional>
#include <memory>
#include <mutex>
#include <queue>
#include <string>
#include <unordered_map>
#include <vector>

namespace estateiq {

// ─── Callbacks ────────────────────────────────────────────────────────────────
using FillCallback   = std::function<void(const Fill&)>;
using RejectCallback = std::function<void(const Order&, const std::string& reason)>;

// ─── Order book (per-symbol) ──────────────────────────────────────────────────
struct OrderBook {
    std::vector<Order> open_orders;
    std::mutex         mtx;
};

// ─── Execution config ─────────────────────────────────────────────────────────
struct ExecutionConfig {
    double commission_rate{0.001};   // 10 bps
    double slippage_bps{5.0};        // 5 bps
    bool   simulate{true};           // paper trading when true
    double max_order_value{1e6};     // single-order value cap
};

// ─── Order Execution Engine ───────────────────────────────────────────────────
class OrderExecutionEngine {
public:
    explicit OrderExecutionEngine(ExecutionConfig cfg = {});

    // Submit an order; returns assigned OrderId
    OrderId submit(Order order);

    // Cancel a pending/open order
    bool cancel(OrderId id);

    // Market-data driven fill simulation (called on each bar)
    void on_bar(const Bar& bar);

    // Register lifecycle callbacks
    void on_fill(FillCallback cb)   { fill_callbacks_.push_back(std::move(cb)); }
    void on_reject(RejectCallback cb) { reject_callbacks_.push_back(std::move(cb)); }

    // Query
    const Order* find_order(OrderId id) const;
    std::vector<Order> open_orders() const;
    std::vector<Fill>  fill_history() const;

    // Stats
    std::size_t total_orders()   const noexcept { return order_counter_.load(); }
    std::size_t total_fills()    const noexcept;
    double      total_commission() const noexcept;

private:
    ExecutionConfig cfg_;
    std::atomic<OrderId> order_counter_{1};
    std::atomic<TradeId> fill_counter_{1};

    mutable std::mutex orders_mtx_;
    std::unordered_map<OrderId, Order> orders_;
    std::unordered_map<std::string, std::vector<OrderId>> symbol_orders_;

    mutable std::mutex fills_mtx_;
    std::vector<Fill> fills_;

    std::vector<FillCallback>   fill_callbacks_;
    std::vector<RejectCallback> reject_callbacks_;

    bool validate(const Order& order, std::string& reason) const;
    Fill simulate_fill(const Order& order, const Bar& bar);
    void emit_fill(const Fill& fill);
    void emit_reject(const Order& order, const std::string& reason);
    void update_order_status(OrderId id, OrderStatus status);
};

}  // namespace estateiq
