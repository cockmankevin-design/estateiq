#pragma once

#include "types.hpp"
#include "order_execution.hpp"
#include "risk_manager.hpp"
#include "portfolio.hpp"

#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace estateiq {

// ─── Strategy interface ───────────────────────────────────────────────────────
class IStrategy {
public:
    virtual ~IStrategy() = default;

    virtual std::string name() const = 0;
    virtual StrategyId  id()   const = 0;

    // Called on each new bar
    virtual std::vector<Signal> on_bar(const Bar& bar) = 0;

    // Called once on start (for initialisation with historical data)
    virtual void on_start(const std::vector<Bar>& history) {}

    // Called when a fill is confirmed
    virtual void on_fill(const Fill& fill) {}
};

// ─── Strategy engine config ───────────────────────────────────────────────────
struct EngineConfig {
    double initial_capital{100'000.0};
    bool   live_mode{false};       // false = backtest / paper
    bool   log_every_bar{false};
};

// ─── Strategy Engine ──────────────────────────────────────────────────────────
class StrategyEngine {
public:
    explicit StrategyEngine(EngineConfig cfg = {});

    // Register components (must be done before run())
    void set_execution(std::shared_ptr<OrderExecutionEngine> exec);
    void set_risk_manager(std::shared_ptr<RiskManager> risk);
    void add_strategy(std::shared_ptr<IStrategy> strategy);

    // Run: feed historical bars through the engine
    void run(const std::vector<Bar>& bars);

    // Live mode: process a single bar (call from market data feed)
    void process_bar(const Bar& bar);

    // Query
    const Portfolio& portfolio() const noexcept { return *portfolio_; }
    Portfolio&       portfolio()       noexcept { return *portfolio_; }

    // Summary
    void print_summary() const;

private:
    EngineConfig cfg_;
    std::shared_ptr<OrderExecutionEngine> exec_;
    std::shared_ptr<RiskManager>         risk_;
    std::unique_ptr<Portfolio>           portfolio_;
    std::vector<std::shared_ptr<IStrategy>> strategies_;

    void dispatch_signals(const std::vector<Signal>& signals,
                          const std::unordered_map<std::string, Price>& prices);
    Order signal_to_order(const Signal& signal,
                          StrategyId sid,
                          const std::unordered_map<std::string, Price>& prices) const;
};

// ─── Built-in strategy implementations ───────────────────────────────────────

// Dual-moving-average crossover
class DualMAStrategy : public IStrategy {
public:
    DualMAStrategy(StrategyId id, std::string symbol,
                   int short_window = 20, int long_window = 200);

    std::string name() const override { return "DualMA"; }
    StrategyId  id()   const override { return id_; }

    std::vector<Signal> on_bar(const Bar& bar) override;

private:
    StrategyId  id_;
    std::string symbol_;
    int         short_w_, long_w_;
    std::vector<Price> closes_;

    double sma(int n) const;
};

// RSI mean-reversion
class RSIMeanReversionStrategy : public IStrategy {
public:
    RSIMeanReversionStrategy(StrategyId id, std::string symbol,
                             int period = 14,
                             double oversold = 30.0,
                             double overbought = 70.0);

    std::string name() const override { return "RSIMeanReversion"; }
    StrategyId  id()   const override { return id_; }

    std::vector<Signal> on_bar(const Bar& bar) override;

private:
    StrategyId  id_;
    std::string symbol_;
    int         period_;
    double      oversold_, overbought_;
    std::vector<Price> closes_;

    double rsi() const;
};

}  // namespace estateiq
