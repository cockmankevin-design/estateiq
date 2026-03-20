#pragma once

#include <chrono>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace estateiq {

// ─── Timestamp ────────────────────────────────────────────────────────────────
using Clock     = std::chrono::system_clock;
using TimePoint = Clock::time_point;
using Duration  = Clock::duration;

// ─── IDs ──────────────────────────────────────────────────────────────────────
using OrderId    = std::uint64_t;
using TradeId    = std::uint64_t;
using StrategyId = std::uint32_t;

// ─── Primitive aliases ────────────────────────────────────────────────────────
using Price    = double;
using Quantity = double;
using Weight   = double;   // portfolio weight [0, 1]

// ─── OHLCV bar ────────────────────────────────────────────────────────────────
struct Bar {
    TimePoint timestamp;
    std::string symbol;
    Price  open;
    Price  high;
    Price  low;
    Price  close;
    double volume;
};

// ─── Order side & type ────────────────────────────────────────────────────────
enum class Side  : std::uint8_t { BUY = 1, SELL = 2 };
enum class OrderType : std::uint8_t {
    MARKET = 1,
    LIMIT  = 2,
    STOP   = 3,
    STOP_LIMIT = 4,
};
enum class OrderStatus : std::uint8_t {
    PENDING  = 0,
    OPEN     = 1,
    FILLED   = 2,
    PARTIAL  = 3,
    CANCELLED = 4,
    REJECTED  = 5,
};

// ─── Order ────────────────────────────────────────────────────────────────────
struct Order {
    OrderId    id{0};
    StrategyId strategy_id{0};
    std::string symbol;
    Side       side;
    OrderType  type;
    Quantity   quantity;
    Price      limit_price{0.0};
    Price      stop_price{0.0};
    TimePoint  created_at;
    TimePoint  updated_at;
    OrderStatus status{OrderStatus::PENDING};
    std::string client_order_id;
};

// ─── Fill / Execution ─────────────────────────────────────────────────────────
struct Fill {
    TradeId    id{0};
    OrderId    order_id{0};
    std::string symbol;
    Side       side;
    Quantity   quantity;
    Price      price;
    double     commission{0.0};
    double     slippage{0.0};
    TimePoint  executed_at;
};

// ─── Position ─────────────────────────────────────────────────────────────────
struct Position {
    std::string symbol;
    Quantity    quantity{0.0};      // positive = long, negative = short
    Price       avg_cost{0.0};      // average entry price
    Price       market_price{0.0};  // latest mark-to-market price
    double      unrealised_pnl{0.0};
    double      realised_pnl{0.0};

    double market_value() const noexcept { return quantity * market_price; }
    bool   is_long()  const noexcept { return quantity > 0.0; }
    bool   is_short() const noexcept { return quantity < 0.0; }
    bool   is_flat()  const noexcept { return quantity == 0.0; }
};

// ─── Signal ───────────────────────────────────────────────────────────────────
enum class SignalType : std::uint8_t { LONG = 1, SHORT = 2, EXIT = 3, HOLD = 4 };

struct Signal {
    TimePoint   timestamp;
    std::string symbol;
    SignalType  type;
    double      confidence{1.0};
    Weight      target_weight{0.0};
    Price       ref_price{0.0};
};

}  // namespace estateiq
