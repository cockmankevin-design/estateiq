/**
 * EstateIQ C++ Core Engine
 *
 * Entry point for:
 *   1. Standalone backtest runs (CSV data input)
 *   2. Live paper/live trading mode (WebSocket feed)
 *   3. Performance benchmarking / smoke test
 */

#include "include/data_loader.hpp"
#include "include/order_execution.hpp"
#include "include/performance.hpp"
#include "include/portfolio.hpp"
#include "include/risk_manager.hpp"
#include "include/strategy_engine.hpp"
#include "include/types.hpp"

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <cmath>
#include <iostream>
#include <memory>
#include <random>
#include <string>
#include <vector>

using namespace estateiq;

// ─── Generate synthetic OHLCV bars (for smoke testing) ───────────────────────
std::vector<Bar> generate_synthetic_bars(
    const std::string& symbol,
    std::size_t n,
    double initial_price = 100.0,
    double drift = 0.0001,
    double vol   = 0.015
) {
    std::mt19937_64 rng(42);
    std::normal_distribution<double> noise(0.0, 1.0);

    std::vector<Bar> bars;
    bars.reserve(n);

    double price = initial_price;
    auto   ts    = Clock::now() - std::chrono::hours(n * 24);

    for (std::size_t i = 0; i < n; ++i) {
        double ret  = drift + vol * noise(rng);
        double open = price;
        price       = open * (1.0 + ret);
        double high = std::max(open, price) * (1.0 + std::abs(noise(rng)) * vol * 0.5);
        double low  = std::min(open, price) * (1.0 - std::abs(noise(rng)) * vol * 0.5);
        double vol_ = 1e6 * (1.0 + std::abs(noise(rng)));

        Bar bar;
        bar.symbol    = symbol;
        bar.timestamp = ts + std::chrono::hours(i * 24);
        bar.open      = open;
        bar.high      = high;
        bar.low       = low;
        bar.close     = price;
        bar.volume    = vol_;
        bars.push_back(bar);
    }
    return bars;
}

// ─── Run backtest demo ────────────────────────────────────────────────────────
void run_demo_backtest() {
    spdlog::info("=== EstateIQ Core Engine — Demo Backtest ===");

    // Generate 5 years of synthetic daily bars
    auto bars = generate_synthetic_bars("DEMO", 252 * 5, 100.0, 0.0003, 0.015);
    spdlog::info("Generated {} synthetic bars for DEMO", bars.size());

    // Execution engine
    ExecutionConfig exec_cfg;
    exec_cfg.commission_rate = 0.001;
    exec_cfg.slippage_bps    = 5.0;
    exec_cfg.simulate        = true;
    auto exec = std::make_shared<OrderExecutionEngine>(exec_cfg);

    // Risk manager
    RiskLimits limits;
    limits.max_position_weight = 1.0;
    limits.max_daily_loss      = 0.05;
    limits.max_drawdown        = 0.25;
    auto risk = std::make_shared<RiskManager>(limits);

    // Strategy engine
    EngineConfig eng_cfg;
    eng_cfg.initial_capital = 100'000.0;
    eng_cfg.live_mode       = false;
    StrategyEngine engine(eng_cfg);
    engine.set_execution(exec);
    engine.set_risk_manager(risk);

    // Add strategies
    engine.add_strategy(std::make_shared<DualMAStrategy>(1, "DEMO", 20, 200));

    // Run
    engine.run(bars);
    engine.print_summary();

    // Compute performance metrics
    auto& nav_hist = engine.portfolio().nav_history();
    PerformanceCalculator perf(0.04);
    auto metrics = perf.compute_from_nav(nav_hist);
    PerformanceCalculator::print(metrics, "Demo Strategy");

    // Output JSON metrics
    std::cout << PerformanceCalculator::to_json(metrics) << '\n';
}

// ─── Main ─────────────────────────────────────────────────────────────────────
int main(int argc, char* argv[]) {
    spdlog::set_level(spdlog::level::info);
    spdlog::set_pattern("[%Y-%m-%d %H:%M:%S.%e] [%^%l%$] %v");

    std::string mode = "demo";
    if (argc > 1) mode = argv[1];

    try {
        if (mode == "demo") {
            run_demo_backtest();
        } else if (mode == "csv" && argc >= 4) {
            // Usage: estateiq_engine csv <path> <symbol>
            DataLoader loader;
            DataSourceConfig cfg;
            cfg.type        = "csv";
            cfg.path_or_url = argv[2];
            cfg.symbol      = argv[3];
            auto bars       = loader.load(cfg);
            spdlog::info("Loaded {} bars for {}", bars.size(), cfg.symbol);

            // Run a simple DualMA backtest on loaded data
            auto exec = std::make_shared<OrderExecutionEngine>();
            auto risk = std::make_shared<RiskManager>();
            StrategyEngine engine;
            engine.set_execution(exec);
            engine.set_risk_manager(risk);
            engine.add_strategy(std::make_shared<DualMAStrategy>(1, cfg.symbol));
            engine.run(bars);
            engine.print_summary();

            auto metrics = PerformanceCalculator(0.04)
                               .compute_from_nav(engine.portfolio().nav_history());
            std::cout << PerformanceCalculator::to_json(metrics) << '\n';
        } else {
            std::cerr << "Usage:\n"
                      << "  estateiq_engine demo\n"
                      << "  estateiq_engine csv <data.csv> <SYMBOL>\n";
            return 1;
        }
    } catch (const std::exception& e) {
        spdlog::critical("Fatal error: {}", e.what());
        return 1;
    }

    return 0;
}
