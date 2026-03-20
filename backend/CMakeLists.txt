cmake_minimum_required(VERSION 3.20)
project(EstateIQCore VERSION 1.0.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# ─── Build options ────────────────────────────────────────────────────────────
option(ENABLE_TESTS  "Build unit tests"          ON)
option(ENABLE_ASAN   "Enable AddressSanitizer"   OFF)
option(ENABLE_TSAN   "Enable ThreadSanitizer"    OFF)

# ─── Compiler flags ───────────────────────────────────────────────────────────
if(MSVC)
    add_compile_options(/W4 /O2)
else()
    add_compile_options(-Wall -Wextra -Wpedantic -O3 -march=native)
    if(ENABLE_ASAN)
        add_compile_options(-fsanitize=address -fno-omit-frame-pointer)
        add_link_options(-fsanitize=address)
    endif()
    if(ENABLE_TSAN)
        add_compile_options(-fsanitize=thread)
        add_link_options(-fsanitize=thread)
    endif()
endif()

# ─── Dependencies ─────────────────────────────────────────────────────────────
find_package(Threads REQUIRED)

# nlohmann/json (header-only, FetchContent fallback)
find_package(nlohmann_json 3.11 QUIET)
if(NOT nlohmann_json_FOUND)
    include(FetchContent)
    FetchContent_Declare(nlohmann_json
        URL https://github.com/nlohmann/json/releases/download/v3.11.3/json.tar.xz)
    FetchContent_MakeAvailable(nlohmann_json)
endif()

# spdlog (logging)
find_package(spdlog QUIET)
if(NOT spdlog_FOUND)
    include(FetchContent)
    FetchContent_Declare(spdlog
        GIT_REPOSITORY https://github.com/gabime/spdlog.git
        GIT_TAG        v1.12.0)
    FetchContent_MakeAvailable(spdlog)
endif()

# ─── Source files ─────────────────────────────────────────────────────────────
set(CORE_SOURCES
    src/order_execution.cpp
    src/risk_manager.cpp
    src/portfolio.cpp
    src/data_loader.cpp
    src/strategy_engine.cpp
    src/performance.cpp
)

# ─── Core shared library ──────────────────────────────────────────────────────
add_library(estateiq_core STATIC ${CORE_SOURCES})
target_include_directories(estateiq_core PUBLIC include)
target_link_libraries(estateiq_core PUBLIC
    Threads::Threads
    nlohmann_json::nlohmann_json
    spdlog::spdlog
)

# ─── Main executable ──────────────────────────────────────────────────────────
add_executable(estateiq_engine main.cpp)
target_link_libraries(estateiq_engine PRIVATE estateiq_core)

# ─── Tests ────────────────────────────────────────────────────────────────────
if(ENABLE_TESTS)
    enable_testing()
    find_package(GTest QUIET)
    if(NOT GTest_FOUND)
        include(FetchContent)
        FetchContent_Declare(googletest
            URL https://github.com/google/googletest/archive/v1.14.0.tar.gz)
        FetchContent_MakeAvailable(googletest)
    endif()

    add_executable(run_tests
        tests/test_order_execution.cpp
        tests/test_risk_manager.cpp
        tests/test_portfolio.cpp
        tests/test_performance.cpp
    )
    target_link_libraries(run_tests PRIVATE estateiq_core GTest::gtest_main)
    include(GoogleTest)
    gtest_discover_tests(run_tests)
endif()
