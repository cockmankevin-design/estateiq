#include "performance.hpp"
#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

#include <algorithm>
#include <cmath>
#include <numeric>
#include <stdexcept>

namespace estateiq {

PerformanceCalculator::PerformanceCalculator(
    double risk_free_rate,
    const std::vector<double>* benchmark_returns
)
    : risk_free_rate_(risk_free_rate)
    , rf_daily_(risk_free_rate / TRADING_DAYS)
    , benchmark_(benchmark_returns)
{}

// ─── Compute all metrics ──────────────────────────────────────────────────────
Metrics PerformanceCalculator::compute(const std::vector<double>& r) const {
    Metrics m;
    if (r.empty()) return m;

    m.n_bars              = r.size();
    m.total_return        = total_return(r);
    m.annualised_return   = annualised_return(r);
    m.cagr                = cagr(r);
    m.annualised_vol      = annualised_vol(r);
    m.max_drawdown        = max_drawdown(r);
    m.sharpe              = sharpe(r, rf_daily_);
    m.sortino             = sortino(r, rf_daily_);
    m.calmar              = calmar(r);
    m.win_rate            = win_rate(r);
    m.profit_factor       = profit_factor(r);
    m.var_95              = var(r, 0.05);
    m.cvar_95             = cvar(r, 0.05);
    m.skewness            = skewness(r);
    m.kurtosis            = kurtosis(r);

    double ag = 0.0;
    for (auto x : r) if (x > 0) ag += x;
    double al = 0.0;
    for (auto x : r) if (x < 0) al += x;
    long n_pos = std::count_if(r.begin(), r.end(), [](double x){ return x > 0; });
    long n_neg = std::count_if(r.begin(), r.end(), [](double x){ return x < 0; });
    m.avg_win  = n_pos > 0 ? ag / n_pos : 0.0;
    m.avg_loss = n_neg > 0 ? al / n_neg : 0.0;

    if (benchmark_ && !benchmark_->empty()) {
        m.beta              = beta(r, *benchmark_);
        m.alpha             = alpha_ann(r, *benchmark_, rf_daily_);
        m.information_ratio = information_ratio(r, *benchmark_);
        m.tracking_error    = tracking_error(r, *benchmark_);
    }
    return m;
}

Metrics PerformanceCalculator::compute_from_nav(
    const std::vector<std::pair<TimePoint, double>>& nav_history
) const {
    std::vector<double> r;
    r.reserve(nav_history.size());
    for (std::size_t i = 1; i < nav_history.size(); ++i) {
        double prev = nav_history[i-1].second;
        double curr = nav_history[i].second;
        r.push_back(prev > 0 ? (curr / prev - 1.0) : 0.0);
    }
    return compute(r);
}

// ─── Static helpers ───────────────────────────────────────────────────────────
double PerformanceCalculator::total_return(const std::vector<double>& r) {
    double cum = 1.0;
    for (auto x : r) cum *= (1.0 + x);
    return cum - 1.0;
}

double PerformanceCalculator::annualised_return(const std::vector<double>& r) {
    double tr = total_return(r);
    double n  = static_cast<double>(r.size());
    return std::pow(1.0 + tr, TRADING_DAYS / n) - 1.0;
}

double PerformanceCalculator::cagr(const std::vector<double>& r) {
    if (r.empty()) return 0.0;
    double n_years = static_cast<double>(r.size()) / TRADING_DAYS;
    double tr      = total_return(r);
    return std::pow(1.0 + tr, 1.0 / std::max(n_years, 1e-6)) - 1.0;
}

double PerformanceCalculator::annualised_vol(const std::vector<double>& r) {
    if (r.size() < 2) return 0.0;
    double mean = std::accumulate(r.begin(), r.end(), 0.0) / r.size();
    double var  = 0.0;
    for (auto x : r) { double d = x - mean; var += d * d; }
    return std::sqrt(var / (r.size() - 1)) * std::sqrt(TRADING_DAYS);
}

double PerformanceCalculator::max_drawdown(const std::vector<double>& r) {
    double cum = 1.0, peak = 1.0, mdd = 0.0;
    for (auto x : r) {
        cum  *= (1.0 + x);
        peak  = std::max(peak, cum);
        mdd   = std::min(mdd, (cum - peak) / peak);
    }
    return mdd;
}

double PerformanceCalculator::sharpe(const std::vector<double>& r, double rf_daily) {
    if (r.size() < 2) return 0.0;
    std::vector<double> excess(r.size());
    for (std::size_t i = 0; i < r.size(); ++i) excess[i] = r[i] - rf_daily;
    double mean = std::accumulate(excess.begin(), excess.end(), 0.0) / excess.size();
    double var  = 0.0;
    for (auto x : excess) { double d = x - mean; var += d * d; }
    double std_dev = std::sqrt(var / (excess.size() - 1));
    return std_dev > 0 ? mean / std_dev * std::sqrt(TRADING_DAYS) : 0.0;
}

double PerformanceCalculator::sortino(const std::vector<double>& r, double rf_daily) {
    if (r.size() < 2) return 0.0;
    double mean_excess = 0.0;
    double down_var    = 0.0;
    int    n_down      = 0;
    for (auto x : r) {
        mean_excess += x - rf_daily;
        if (x < rf_daily) { double d = x - rf_daily; down_var += d * d; ++n_down; }
    }
    mean_excess /= r.size();
    if (n_down < 2) return 0.0;
    double down_std = std::sqrt(down_var / n_down) * std::sqrt(TRADING_DAYS);
    return down_std > 0 ? mean_excess * TRADING_DAYS / down_std : 0.0;
}

double PerformanceCalculator::calmar(const std::vector<double>& r) {
    double mdd = std::abs(max_drawdown(r));
    return mdd > 0 ? cagr(r) / mdd : 0.0;
}

double PerformanceCalculator::win_rate(const std::vector<double>& r) {
    if (r.empty()) return 0.0;
    long wins = std::count_if(r.begin(), r.end(), [](double x){ return x > 0; });
    return static_cast<double>(wins) / r.size();
}

double PerformanceCalculator::profit_factor(const std::vector<double>& r) {
    double gross_win = 0.0, gross_loss = 0.0;
    for (auto x : r) { if (x > 0) gross_win += x; else gross_loss -= x; }
    return gross_loss > 0 ? gross_win / gross_loss : 1e9;
}

double PerformanceCalculator::var(const std::vector<double>& r, double alpha) {
    if (r.empty()) return 0.0;
    std::vector<double> sorted = r;
    std::sort(sorted.begin(), sorted.end());
    std::size_t idx = static_cast<std::size_t>(alpha * sorted.size());
    return sorted[std::min(idx, sorted.size() - 1)];
}

double PerformanceCalculator::cvar(const std::vector<double>& r, double alpha) {
    if (r.empty()) return 0.0;
    double cutoff = var(r, alpha);
    double sum = 0.0; int cnt = 0;
    for (auto x : r) if (x <= cutoff) { sum += x; ++cnt; }
    return cnt > 0 ? sum / cnt : cutoff;
}

double PerformanceCalculator::skewness(const std::vector<double>& r) {
    if (r.size() < 3) return 0.0;
    double n    = r.size();
    double mean = std::accumulate(r.begin(), r.end(), 0.0) / n;
    double m2   = 0.0, m3 = 0.0;
    for (auto x : r) {
        double d = x - mean;
        m2 += d * d;
        m3 += d * d * d;
    }
    m2 /= n; m3 /= n;
    double sigma = std::sqrt(m2);
    return sigma > 0 ? m3 / (sigma * sigma * sigma) : 0.0;
}

double PerformanceCalculator::kurtosis(const std::vector<double>& r) {
    if (r.size() < 4) return 0.0;
    double n    = r.size();
    double mean = std::accumulate(r.begin(), r.end(), 0.0) / n;
    double m2   = 0.0, m4 = 0.0;
    for (auto x : r) {
        double d = x - mean;
        m2 += d * d;
        m4 += d * d * d * d;
    }
    m2 /= n; m4 /= n;
    return m2 > 0 ? m4 / (m2 * m2) - 3.0 : 0.0;  // excess kurtosis
}

double PerformanceCalculator::beta(const std::vector<double>& r, const std::vector<double>& b) {
    std::size_t n = std::min(r.size(), b.size());
    if (n < 2) return 1.0;
    double mr = 0.0, mb = 0.0;
    for (std::size_t i = 0; i < n; ++i) { mr += r[i]; mb += b[i]; }
    mr /= n; mb /= n;
    double cov = 0.0, var_b = 0.0;
    for (std::size_t i = 0; i < n; ++i) {
        cov   += (r[i] - mr) * (b[i] - mb);
        var_b += (b[i] - mb) * (b[i] - mb);
    }
    return var_b > 0 ? cov / var_b : 0.0;
}

double PerformanceCalculator::alpha_ann(const std::vector<double>& r,
                                        const std::vector<double>& b,
                                        double rf_daily) {
    double b_ = beta(r, b);
    std::size_t n = std::min(r.size(), b.size());
    double mr = 0.0, mb = 0.0;
    for (std::size_t i = 0; i < n; ++i) { mr += r[i]; mb += b[i]; }
    mr /= n; mb /= n;
    double alpha_daily = mr - rf_daily - b_ * (mb - rf_daily);
    return alpha_daily * TRADING_DAYS;
}

double PerformanceCalculator::information_ratio(const std::vector<double>& r,
                                                 const std::vector<double>& b) {
    std::size_t n = std::min(r.size(), b.size());
    if (n < 2) return 0.0;
    std::vector<double> active(n);
    for (std::size_t i = 0; i < n; ++i) active[i] = r[i] - b[i];
    double mean = std::accumulate(active.begin(), active.end(), 0.0) / n;
    double var  = 0.0;
    for (auto x : active) { double d = x - mean; var += d * d; }
    double te = std::sqrt(var / (n - 1)) * std::sqrt(TRADING_DAYS);
    return te > 0 ? mean * TRADING_DAYS / te : 0.0;
}

double PerformanceCalculator::tracking_error(const std::vector<double>& r,
                                              const std::vector<double>& b) {
    std::size_t n = std::min(r.size(), b.size());
    if (n < 2) return 0.0;
    std::vector<double> active(n);
    for (std::size_t i = 0; i < n; ++i) active[i] = r[i] - b[i];
    double mean = std::accumulate(active.begin(), active.end(), 0.0) / n;
    double var  = 0.0;
    for (auto x : active) { double d = x - mean; var += d * d; }
    return std::sqrt(var / (n - 1)) * std::sqrt(TRADING_DAYS);
}

// ─── Print ────────────────────────────────────────────────────────────────────
void PerformanceCalculator::print(const Metrics& m, const std::string& title) {
    spdlog::info("========== {} ==========", title);
    spdlog::info("Total Return      : {:.2f}%", m.total_return        * 100);
    spdlog::info("CAGR              : {:.2f}%", m.cagr                * 100);
    spdlog::info("Annualised Vol    : {:.2f}%", m.annualised_vol      * 100);
    spdlog::info("Max Drawdown      : {:.2f}%", m.max_drawdown        * 100);
    spdlog::info("Sharpe Ratio      : {:.3f}",  m.sharpe);
    spdlog::info("Sortino Ratio     : {:.3f}",  m.sortino);
    spdlog::info("Calmar Ratio      : {:.3f}",  m.calmar);
    spdlog::info("Win Rate          : {:.2f}%", m.win_rate            * 100);
    spdlog::info("Profit Factor     : {:.3f}",  m.profit_factor);
    spdlog::info("VaR 95%           : {:.2f}%", m.var_95              * 100);
    spdlog::info("CVaR 95%          : {:.2f}%", m.cvar_95             * 100);
    spdlog::info("Alpha             : {:.2f}%", m.alpha               * 100);
    spdlog::info("Beta              : {:.3f}",  m.beta);
    spdlog::info("Information Ratio : {:.3f}",  m.information_ratio);
    spdlog::info("Tracking Error    : {:.2f}%", m.tracking_error      * 100);
    spdlog::info("==================================");
}

// ─── JSON serialisation ───────────────────────────────────────────────────────
std::string PerformanceCalculator::to_json(const Metrics& m) {
    nlohmann::json j = {
        {"total_return",       m.total_return},
        {"annualised_return",  m.annualised_return},
        {"cagr",               m.cagr},
        {"annualised_vol",     m.annualised_vol},
        {"max_drawdown",       m.max_drawdown},
        {"sharpe",             m.sharpe},
        {"sortino",            m.sortino},
        {"calmar",             m.calmar},
        {"win_rate",           m.win_rate},
        {"profit_factor",      m.profit_factor},
        {"avg_win",            m.avg_win},
        {"avg_loss",           m.avg_loss},
        {"var_95",             m.var_95},
        {"cvar_95",            m.cvar_95},
        {"skewness",           m.skewness},
        {"kurtosis",           m.kurtosis},
        {"alpha",              m.alpha},
        {"beta",               m.beta},
        {"information_ratio",  m.information_ratio},
        {"tracking_error",     m.tracking_error},
        {"n_trades",           m.n_trades},
        {"n_bars",             m.n_bars},
    };
    return j.dump(2);
}

}  // namespace estateiq
