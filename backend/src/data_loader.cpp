#include "data_loader.hpp"
#include <spdlog/spdlog.h>

#include <algorithm>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <sstream>
#include <stdexcept>

namespace estateiq {

// ─── Public interface ─────────────────────────────────────────────────────────
std::vector<Bar> DataLoader::load(const DataSourceConfig& cfg) const {
    if (cfg.type == "csv") {
        return load_csv(cfg.path_or_url, cfg.symbol, cfg.csv_delimiter, cfg.has_header);
    }
    if (cfg.type == "binary") {
        return load_binary(cfg.path_or_url);
    }
    throw std::invalid_argument("Unsupported data source type: " + cfg.type);
}

void DataLoader::load_streaming(
    const DataSourceConfig& cfg,
    std::function<void(const Bar&)> handler
) const {
    auto bars = load(cfg);
    for (auto& bar : bars) handler(bar);
}

// ─── CSV ──────────────────────────────────────────────────────────────────────
std::vector<Bar> DataLoader::load_csv(
    const std::string& path,
    const std::string& symbol,
    char delim,
    bool header
) const {
    std::ifstream file(path);
    if (!file.is_open())
        throw std::runtime_error("Cannot open file: " + path);

    std::vector<Bar> bars;
    std::string line;
    bool first = true;

    while (std::getline(file, line)) {
        if (line.empty()) continue;
        if (first && header) { first = false; continue; }
        first = false;
        try {
            bars.push_back(parse_csv_row(line, symbol, delim, true));
        } catch (std::exception& e) {
            spdlog::warn("Skipping malformed CSV row: {} — {}", line, e.what());
        }
    }

    sort_bars(bars);
    spdlog::info("Loaded {} bars for {} from {}", bars.size(), symbol, path);
    return bars;
}

Bar DataLoader::parse_csv_row(
    const std::string& line,
    const std::string& symbol,
    char delim,
    bool /*has_date_col*/
) const {
    std::vector<std::string> fields;
    std::string token;
    std::istringstream ss(line);
    while (std::getline(ss, token, delim)) {
        // Strip quotes
        if (!token.empty() && token.front() == '"') token = token.substr(1);
        if (!token.empty() && token.back()  == '"') token.pop_back();
        fields.push_back(token);
    }

    // Expected columns: date, open, high, low, close[, volume]
    if (fields.size() < 5)
        throw std::runtime_error("Too few columns");

    Bar bar;
    bar.symbol    = symbol;
    bar.timestamp = parse_timestamp(fields[0]);
    bar.open      = std::stod(fields[1]);
    bar.high      = std::stod(fields[2]);
    bar.low       = std::stod(fields[3]);
    bar.close     = std::stod(fields[4]);
    bar.volume    = fields.size() > 5 ? std::stod(fields[5]) : 0.0;
    return bar;
}

TimePoint DataLoader::parse_timestamp(const std::string& s) const {
    std::tm tm{};
    std::istringstream ss(s);
    // Try ISO 8601: YYYY-MM-DD
    ss >> std::get_time(&tm, "%Y-%m-%d");
    if (ss.fail()) {
        // Try MM/DD/YYYY
        ss.clear(); ss.str(s);
        ss >> std::get_time(&tm, "%m/%d/%Y");
    }
    if (ss.fail()) throw std::runtime_error("Cannot parse date: " + s);
    tm.tm_isdst = -1;
    return Clock::from_time_t(std::mktime(&tm));
}

// ─── Binary (custom packed format) ───────────────────────────────────────────
// Format per bar: [int64 epoch_ms][8×double: open,high,low,close,volume,0,0,0]
// symbol stored in file header: [uint16 sym_len][bytes]
static constexpr std::size_t BAR_SIZE = sizeof(std::int64_t) + 8 * sizeof(double);

std::vector<Bar> DataLoader::load_binary(const std::string& path) const {
    std::ifstream file(path, std::ios::binary);
    if (!file.is_open()) throw std::runtime_error("Cannot open binary file: " + path);

    // Read symbol header
    std::uint16_t sym_len = 0;
    file.read(reinterpret_cast<char*>(&sym_len), sizeof(sym_len));
    std::string symbol(sym_len, '\0');
    file.read(symbol.data(), sym_len);

    std::vector<Bar> bars;
    while (file) {
        std::int64_t epoch_ms = 0;
        double fields[8] = {};
        if (!file.read(reinterpret_cast<char*>(&epoch_ms), sizeof(epoch_ms))) break;
        if (!file.read(reinterpret_cast<char*>(fields), sizeof(fields))) break;

        Bar bar;
        bar.symbol    = symbol;
        bar.timestamp = Clock::time_point(std::chrono::milliseconds(epoch_ms));
        bar.open      = fields[0];
        bar.high      = fields[1];
        bar.low       = fields[2];
        bar.close     = fields[3];
        bar.volume    = fields[4];
        bars.push_back(bar);
    }

    spdlog::info("Loaded {} bars for {} from binary {}", bars.size(), symbol, path);
    return bars;
}

bool DataLoader::save_binary(const std::string& path, const std::vector<Bar>& bars) const {
    if (bars.empty()) return true;
    std::ofstream file(path, std::ios::binary | std::ios::trunc);
    if (!file.is_open()) return false;

    const std::string& sym = bars[0].symbol;
    std::uint16_t sym_len  = static_cast<std::uint16_t>(sym.size());
    file.write(reinterpret_cast<const char*>(&sym_len), sizeof(sym_len));
    file.write(sym.data(), sym_len);

    for (auto& bar : bars) {
        auto epoch_ms = std::chrono::duration_cast<std::chrono::milliseconds>(
            bar.timestamp.time_since_epoch()).count();
        double fields[8] = {bar.open, bar.high, bar.low, bar.close, bar.volume};
        file.write(reinterpret_cast<const char*>(&epoch_ms), sizeof(epoch_ms));
        file.write(reinterpret_cast<const char*>(fields), sizeof(fields));
    }
    return true;
}

bool DataLoader::save_csv(const std::string& path, const std::vector<Bar>& bars) const {
    std::ofstream file(path);
    if (!file.is_open()) return false;
    file << "date,open,high,low,close,volume\n";
    for (auto& bar : bars) {
        auto t = Clock::to_time_t(bar.timestamp);
        std::tm* tm = std::gmtime(&t);
        file << std::put_time(tm, "%Y-%m-%d") << ','
             << bar.open << ',' << bar.high << ',' << bar.low << ','
             << bar.close << ',' << bar.volume << '\n';
    }
    return true;
}

// ─── Utilities ────────────────────────────────────────────────────────────────
void DataLoader::sort_bars(std::vector<Bar>& bars) {
    std::sort(bars.begin(), bars.end(),
              [](const Bar& a, const Bar& b) { return a.timestamp < b.timestamp; });
}

bool DataLoader::validate_bars(const std::vector<Bar>& bars, std::string& error) {
    for (std::size_t i = 0; i < bars.size(); ++i) {
        auto& b = bars[i];
        if (b.high < b.low) {
            error = "Bar " + std::to_string(i) + ": high < low";
            return false;
        }
        if (b.close < 0 || b.open < 0) {
            error = "Bar " + std::to_string(i) + ": negative price";
            return false;
        }
        if (i > 0 && bars[i].timestamp <= bars[i-1].timestamp) {
            error = "Bars not sorted at index " + std::to_string(i);
            return false;
        }
    }
    return true;
}

}  // namespace estateiq
