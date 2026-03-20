#pragma once

#include "types.hpp"

#include <functional>
#include <string>
#include <vector>

namespace estateiq {

// ─── Data source config ───────────────────────────────────────────────────────
struct DataSourceConfig {
    std::string type;          // "csv" | "binary" | "database"
    std::string path_or_url;
    std::string symbol;
    std::string start_date;    // YYYY-MM-DD
    std::string end_date;
    std::string interval;      // "1d" | "1h" | "1m"
    char csv_delimiter{','};
    bool has_header{true};
};

// ─── Data Loader ──────────────────────────────────────────────────────────────
class DataLoader {
public:
    DataLoader() = default;

    // Bulk load: returns all bars for a symbol and date range
    std::vector<Bar> load(const DataSourceConfig& cfg) const;

    // Streaming load: calls handler for each bar as it is read
    void load_streaming(
        const DataSourceConfig& cfg,
        std::function<void(const Bar&)> handler
    ) const;

    // Convenience helpers
    std::vector<Bar> load_csv(const std::string& path, const std::string& symbol,
                               char delim = ',', bool header = true) const;
    std::vector<Bar> load_binary(const std::string& path) const;

    // Write helpers (for caching preprocessed data)
    bool save_binary(const std::string& path, const std::vector<Bar>& bars) const;
    bool save_csv(const std::string& path, const std::vector<Bar>& bars) const;

    // Sort and validate a bar vector
    static void sort_bars(std::vector<Bar>& bars);
    static bool validate_bars(const std::vector<Bar>& bars, std::string& error);

private:
    Bar parse_csv_row(const std::string& line, const std::string& symbol,
                      char delim, bool has_date_col) const;
    TimePoint parse_timestamp(const std::string& s) const;
};

}  // namespace estateiq
