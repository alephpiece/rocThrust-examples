#ifndef TINY_TIMER_H_
#define TINY_TIMER_H_

#include <chrono>
#include <sstream>
#include <stack>

#include "log_utils.h"  /* namespace logutils */
#include "tinyrecord.h" /* TinyRecord */


///-----------------------------------------------------------------------------
/// \class TinyTimer
/// \brief A tiny timer with a resolution of 1 us
/// \param T_Clock  Clock type
///-----------------------------------------------------------------------------
template <typename T_Clock = std::chrono::steady_clock>
class TinyTimer {

protected:

    using duration_type   = std::chrono::microseconds;
    using time_point_type = std::chrono::time_point<T_Clock>;

    const size_t _align_left  = 40;
    const size_t _align_right = 15;

    ///< Starting time points
    std::stack<time_point_type> _start_times;

    ///< Records
    TinyRecord<std::string, duration_type> _records;


public:

    TinyTimer() = default;

    virtual ~TinyTimer() {
        if (!_start_times.empty()) {
            logutils::print("Timer: {} starting points didn't be consumed");
        }
    }


    /// \brief Start a record
    void start() {

        // Record the starting time point
        _start_times.push(T_Clock::now());
    }


    /// \brief Stop and record duration relative to the nearest start
    /// \name Record name
    void stop(const std::string &name) {

        // Compute the duration
        auto end_time = T_Clock::now();

        duration_type time_elapsed;

        // Check if the stack is empty. Each stop time must match
        // a starting time stored in the stack.
        if (!_start_times.empty()) {

            time_elapsed =
                std::chrono::duration_cast<duration_type>(end_time - _start_times.top());

            _start_times.pop();
        }

        append(name, time_elapsed);
    }


    /// \brief Clear records
    virtual void clear() {
        _records.clear();
    }


    /// \brief Print a record
    /// \param name Record name
    void print(const std::string &name) {

        fmt::print("{}\n", this->format(name));
    }


    /// \brief Print all records
    void report() {

        std::ostringstream ss;

        for (const auto &record : _records)
            ss << this->format(record.first) << '\n';

        fmt::print("{}", ss.str());
    }


protected:

    /// \brief Append a record to the map
    /// \param name     Record name
    /// \param duration Elapsed time
    void append(const std::string &name, const duration_type &duration) {

        // Check if the record name exists. Values of elapsed times
        // which have the same record name will be accumulated.
        if (_records.find(name) != _records.end())
            _records.at(name) += duration;
        else
            _records.insert(name, duration);
    }


    /// \brief Get a duration by record name
    /// \param name Record name
    duration_type get(const std::string &name) {

        if (_records.find(name) == _records.end())
            return duration_type();
        else
            return _records.at(name);
    }


    /// \brief Format a record
    /// \param name Record name
    std::string format(const std::string &name) {

        auto us =
            std::chrono::duration_cast<std::chrono::microseconds>( get(name) ).count();

        double ms_ticks = us / 1.0e3;

        return logutils::format("{:.<{}}{:.>{}.2f} ms", name, _align_left, ms_ticks, _align_right);
    }

};

#endif  // TINY_TIMER_H_
