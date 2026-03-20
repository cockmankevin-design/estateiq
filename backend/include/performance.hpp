#pragma once

#include "types.hpp"

#include <string>
#include <unordered_map>
#include <vector>

namespace estateiq {

// ─── Performance metrics ──────────────────────────────────────────────────────
struct Metrics {
    double total_return{0.0};
    double annualised_return{0.0};
    double cagr{0.0};
    double annualised_vol{0.0};
    double max_drawdown{0.0};
    double sharpe{0.0};
    double sortino{0.0};
    double calmar{0.0};
    double win_rate{0.0};
    double profit_factor{0.0};
    double avg_win{0.0};
    double avg_loss{0.0};
    double var_95{0.0};
    double cvar_95{0.0};
    double skewness{0.0};
    double kurtosis{0.0};
    // Benchmark-relative (when benchmark provided)
    double alpha{0.0};
    double beta{0.0};
    double information_ratio{0.0};
    double tracking_error{0.0};
    // Counts
    std::size_t n_trades{0};
    std::size_t n_bars{0};
};

// ─── Performance Calculator ───────────────────────────────────────────────────
class PerformanceCalculator {
public:
    static constexpr int TRADING_DAYS = 252;

    explicit PerformanceCalculator(
        double risk_free_rate = 0.04,
        const std::vector<double>* benchmark_returns = nullptr
    );

    // Compute all metrics from a returns series
    Metrics compute(const std::vector<double>& daily_returns) const;

    // Compute from NAV history
    Metrics compute_from_nav(
        const std::vector<std::pair<TimePoint, double>>& nav_history
    ) const;

    // Individual metric helpers (public for unit testing)
    static double total_return(const std::vector<double>& r);
    static double annualised_return(const std::vector<double>& r);
    static double cagr(const std::vector<double>& r);
    static double annualised_vol(const std::vector<double>& r);
    static double max_drawdown(const std::vector<double>& r);
    static double sharpe(const std::vector<double>& r, double rf_daily);
    static double sortino(const std::vector<double>& r, double rf_daily);
    static double calmar(const std::vector<double>& r);
    static double win_rate(const std::vector<double>& r);
    static double profit_factor(const std::vector<double>& r);
    static double var(const std::vector<double>& r, double alpha = 0.05);
    static double cvar(const std::vector<double>& r, double alpha = 0.05);
    static double skewness(const std::vector<double>& r);
    static double kurtosis(const std::vector<double>& r);
    static double beta(const std::vector<double>& r, const std::vector<double>& b);
    static double alpha_ann(const std::vector<double>& r,
                            const std::vector<double>& b,
                            double rf_daily);
    static double information_ratio(const std::vector<double>& r,
                                    const std::vector<double>& b);
    static double tracking_error(const std::vector<double>& r,
                                 const std::vector<double>& b);

    // Pretty-print metrics to stdout
    static void print(const Metrics& m, const std::string& title = "Performance");

    // Serialise to JSON string (requires nlohmann/json)
    static std::string to_json(const Metrics& m);

private:
    double risk_free_rate_;
    double rf_daily_;
    const std::vector<double>* benchmark_;
};

}  // namespace estateiq
