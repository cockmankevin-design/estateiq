#include "strategy_engine.hpp"
#include <spdlog/spdlog.h>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace estateiq {

// ─── StrategyEngine ───────────────────────────────────────────────────────────
StrategyEngine::StrategyEngine(EngineConfig cfg)
    : cfg_(std::move(cfg))
    , portfolio_(std::make_unique<Portfolio>(cfg_.initial_capital))
{}

void StrategyEngine::set_execution(std::shared_ptr<OrderExecutionEngine> exec) {
    exec_ = std::move(exec);
    exec_->on_fill([this](const Fill& fill) {
        portfolio_->apply_fill(fill);
        for (auto& strat : strategies_) strat->on_fill(fill);
    });
}

void StrategyEngine::set_risk_manager(std::shared_ptr<RiskManager> risk) {
    risk_ = std::move(risk);
}

void StrategyEngine::add_strategy(std::shared_ptr<IStrategy> strategy) {
    strategies_.push_back(std::move(strategy));
}

// ─── Backtest run ─────────────────────────────────────────────────────────────
void StrategyEngine::run(const std::vector<Bar>& bars) {
    if (!exec_) throw std::runtime_error("No execution engine set");

    // Initialise strategies with available history
    for (auto& strat : strategies_) strat->on_start(bars);

    spdlog::info("Backtest start: {} bars, {} strategies, capital={:.0f}",
                 bars.size(), strategies_.size(), cfg_.initial_capital);

    for (auto& bar : bars) {
        process_bar(bar);
    }

    spdlog::info("Backtest complete. Final NAV={:.2f}", portfolio_->nav());
}

// ─── Process single bar ───────────────────────────────────────────────────────
void StrategyEngine::process_bar(const Bar& bar) {
    if (!exec_) return;

    // Feed bar to execution engine (fills pending orders)
    exec_->on_bar(bar);

    // Update portfolio mark-to-market
    portfolio_->mark_to_market({{bar.symbol, bar.close}});
    portfolio_->record_nav(bar.timestamp);

    // Update risk manager
    if (risk_) {
        double nav       = portfolio_->nav();
        double daily_pnl = portfolio_->realised_pnl() + portfolio_->unrealised_pnl();
        risk_->update(nav, daily_pnl, portfolio_->peak_nav());

        if (risk_->is_halted()) {
            spdlog::warn("Trading halted on {}", bar.symbol);
            return;
        }
    }

    // Generate signals from each strategy
    std::unordered_map<std::string, Price> prices{{bar.symbol, bar.close}};
    for (auto& strat : strategies_) {
        auto signals = strat->on_bar(bar);
        if (!signals.empty()) {
            dispatch_signals(signals, prices);
        }
    }

    if (cfg_.log_every_bar) {
        spdlog::debug("Bar processed: {} close={:.4f} NAV={:.2f}",
                      bar.symbol, bar.close, portfolio_->nav());
    }
}

void StrategyEngine::dispatch_signals(
    const std::vector<Signal>& signals,
    const std::unordered_map<std::string, Price>& prices
) {
    auto snap = portfolio_->snapshot();
    for (auto& sig : signals) {
        if (risk_) {
            auto check = risk_->check_signal(sig, snap.positions, snap.nav);
            if (!check.passed) {
                spdlog::warn("Signal rejected for {}: {}", sig.symbol, check.violation);
                continue;
            }
        }
        if (sig.type == SignalType::HOLD) continue;

        Order order = signal_to_order(sig, 0, prices);
        if (order.quantity <= 0) continue;

        if (risk_) {
            auto px_it = prices.find(sig.symbol);
            Price px   = px_it != prices.end() ? px_it->second : 0.0;
            auto check = risk_->check_order(order, snap.positions, snap.nav, px);
            if (!check.passed) {
                spdlog::warn("Order rejected for {}: {}", sig.symbol, check.violation);
                continue;
            }
        }
        exec_->submit(order);
    }
}

Order StrategyEngine::signal_to_order(
    const Signal& signal,
    StrategyId sid,
    const std::unordered_map<std::string, Price>& prices
) const {
    Order order;
    order.strategy_id = sid;
    order.symbol      = signal.symbol;
    order.type        = OrderType::MARKET;
    order.created_at  = signal.timestamp;

    double nav      = portfolio_->nav();
    auto price_it   = prices.find(signal.symbol);
    Price px        = price_it != prices.end() ? price_it->second : signal.ref_price;
    if (px <= 0) return order;   // can't size without price

    double target_val = nav * signal.target_weight;
    auto pos = portfolio_->position(signal.symbol);
    double current_val = pos ? pos->market_value() : 0.0;

    if (signal.type == SignalType::LONG) {
        order.side     = Side::BUY;
        double delta   = target_val - std::max(0.0, current_val);
        order.quantity = std::max(0.0, delta / px);
    } else if (signal.type == SignalType::SHORT) {
        order.side     = Side::SELL;
        double delta   = target_val + std::min(0.0, current_val);
        order.quantity = std::max(0.0, delta / px);
    } else if (signal.type == SignalType::EXIT) {
        order.side     = pos && pos->is_long() ? Side::SELL : Side::BUY;
        order.quantity = pos ? std::abs(pos->quantity) : 0.0;
    }
    return order;
}

void StrategyEngine::print_summary() const {
    spdlog::info("=== Strategy Engine Summary ===");
    spdlog::info("Initial capital : {:.2f}", cfg_.initial_capital);
    spdlog::info("Final NAV       : {:.2f}", portfolio_->nav());
    spdlog::info("Realised P&L    : {:.2f}", portfolio_->realised_pnl());
    spdlog::info("Unrealised P&L  : {:.2f}", portfolio_->unrealised_pnl());
    spdlog::info("Open positions  : {}", portfolio_->snapshot().positions.size());
    spdlog::info("Total fills     : {}", exec_ ? exec_->total_fills() : 0);
    spdlog::info("Total commission: {:.2f}", exec_ ? exec_->total_commission() : 0.0);
}

// ─── DualMAStrategy ───────────────────────────────────────────────────────────
DualMAStrategy::DualMAStrategy(StrategyId id, std::string symbol,
                                int short_window, int long_window)
    : id_(id), symbol_(std::move(symbol))
    , short_w_(short_window), long_w_(long_window) {}

std::vector<Signal> DualMAStrategy::on_bar(const Bar& bar) {
    if (bar.symbol != symbol_) return {};
    closes_.push_back(bar.close);
    if (static_cast<int>(closes_.size()) < long_w_) return {};

    double s = sma(short_w_);
    double l = sma(long_w_);

    Signal sig;
    sig.timestamp    = bar.timestamp;
    sig.symbol       = symbol_;
    sig.ref_price    = bar.close;
    sig.confidence   = std::min(1.0, std::abs(s - l) / l * 10.0);
    sig.target_weight = 1.0;

    if (s > l) {
        sig.type = SignalType::LONG;
        return {sig};
    } else if (s < l) {
        sig.type = SignalType::EXIT;
        return {sig};
    }
    return {};
}

double DualMAStrategy::sma(int n) const {
    int sz = static_cast<int>(closes_.size());
    if (sz < n) return 0.0;
    double sum = 0.0;
    for (int i = sz - n; i < sz; ++i) sum += closes_[i];
    return sum / n;
}

// ─── RSIMeanReversionStrategy ─────────────────────────────────────────────────
RSIMeanReversionStrategy::RSIMeanReversionStrategy(
    StrategyId id, std::string symbol,
    int period, double oversold, double overbought
) : id_(id), symbol_(std::move(symbol))
  , period_(period), oversold_(oversold), overbought_(overbought) {}

std::vector<Signal> RSIMeanReversionStrategy::on_bar(const Bar& bar) {
    if (bar.symbol != symbol_) return {};
    closes_.push_back(bar.close);
    if (static_cast<int>(closes_.size()) <= period_) return {};

    double r = rsi();
    Signal sig;
    sig.timestamp    = bar.timestamp;
    sig.symbol       = symbol_;
    sig.ref_price    = bar.close;
    sig.target_weight = 1.0;

    if (r < oversold_) {
        sig.type       = SignalType::LONG;
        sig.confidence = (oversold_ - r) / oversold_;
        return {sig};
    } else if (r > overbought_) {
        sig.type       = SignalType::SHORT;
        sig.confidence = (r - overbought_) / (100.0 - overbought_);
        return {sig};
    }
    return {};
}

double RSIMeanReversionStrategy::rsi() const {
    int sz = static_cast<int>(closes_.size());
    double avg_gain = 0.0, avg_loss = 0.0;
    for (int i = sz - period_; i < sz; ++i) {
        double diff = closes_[i] - closes_[i - 1];
        if (diff > 0) avg_gain += diff;
        else          avg_loss -= diff;
    }
    avg_gain /= period_;
    avg_loss /= period_;
    if (avg_loss < 1e-12) return 100.0;
    double rs = avg_gain / avg_loss;
    return 100.0 - 100.0 / (1.0 + rs);
}

}  // namespace estateiq
