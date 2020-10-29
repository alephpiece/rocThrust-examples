#ifndef GPU_TINY_TIMER_H_
#define GPU_TINY_TIMER_H_

#include "hip/hip_runtime.h"
#include <deque>
#include <stdexcept>
#include "tinytimer.h"          /* TinyTimer */


///-----------------------------------------------------------------------------
/// \class GPUTinyTimer
/// \brief A tiny timer for gpu
///-----------------------------------------------------------------------------
class GPUTinyTimer: public TinyTimer<std::chrono::steady_clock> {

private:

    ///< Streams
    std::stack<hipStream_t> _streams;

    ///< Event stack
    std::stack<hipEvent_t> _start_events;

    ///< Record names to event pairs
    TinyRecord<std::string, std::pair<hipEvent_t, hipEvent_t>> _name_to_events;

public:

    GPUTinyTimer() = default;

    ~GPUTinyTimer() override {
        clear();
    }


    /// \brief Record an event in a stream as the start
    void start(hipStream_t stream = nullptr) {
        return;

        // Create an event
        hipEvent_t e_start;
        hipEventCreate(&e_start);

        // Record the event
        hipEventRecord(e_start, stream);

        _start_events.push(e_start);
        _streams.push(stream);
    }


    /// \brief Require the timer to put a timestamp at the current position in the stream
    void stop(const std::string &name) {
        return;

        // Create an event
        hipEvent_t e_stop;
        hipEventCreate(&e_stop);

        if (_streams.empty() || _start_events.empty()) {
            throw std::logic_error(fmt::format("Mismatched (start, stop) for timer at {}\n", name));
        }
        else {

            auto stream = _streams.top();
            _streams.pop();

            auto e_start = _start_events.top();
            _start_events.pop();

            // Record the event
            hipEventRecord(e_stop, stream);

            // Store a pair of events for future use
            _name_to_events.insert(name, {e_start, e_stop});

            // Add an empty record
            append(name, duration_type());
        }
    }


    /// \brief Synchronize all of the events and compute the duration
    /// \param Events will be removed after synchronization
    void synchronize() {
        return;

        if (_name_to_events.size() != _records.size()) {
            throw std::logic_error("Mismatched record name and event pair");
        }
        else {

            // Synchronize all events and update records
            for (auto &p : _name_to_events) {
                auto &name    = p.first;
                auto &e_start = p.second.first;
                auto &e_stop  = p.second.second;

                hipEventSynchronize(e_stop);

                // Compute the duration
                float event_ms = 0.;
                hipEventElapsedTime(&event_ms, e_start, e_stop);

                auto time_elapsed =
                    std::chrono::duration_cast<duration_type>(
                        std::chrono::microseconds( int(event_ms * 1e3)) );

                append(name, time_elapsed);
            }
            this->clear();
        }
    }


    void clear() override {
        while (!_start_events.empty()) {
            hipEventDestroy(_start_events.top());
            _start_events.pop();
        }

        for (auto &p : _name_to_events) {
            hipEventDestroy(p.second.first);
            hipEventDestroy(p.second.second);
        }
        _name_to_events.clear();
    }

};

#endif  // GPU_TINY_TIMER_H_
