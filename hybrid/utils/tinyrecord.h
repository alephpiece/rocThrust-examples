#ifndef TINY_RECORD_H_
#define TINY_RECORD_H_

#include <algorithm>
#include <stdexcept>
#include <utility>
#include <vector>


///-----------------------------------------------------------------------------
/// \class TinyRecord
/// \brief A map for timer records
///-----------------------------------------------------------------------------
template <typename K, typename V>
class TinyRecord {

public:

    using key_type = K;
    using value_type = V;

    ///< A vector of pairs
    using VectorType = std::vector<std::pair<key_type, value_type>>;

    ///< Data
    VectorType _data;

    TinyRecord() = default;
    virtual ~TinyRecord() = default;

    auto begin() noexcept -> decltype(_data.begin()) { return _data.begin(); }
    auto end() noexcept -> decltype(_data.end())     { return _data.end(); }

    /// \brief   Find a specific key
    /// \details Modify this method to improve performance if necessary
    auto find(const key_type &key) -> decltype(_data.end()) {
        return std::find_if(this->begin(), this->end(),
                [&key](const std::pair<key_type, value_type> &e) {
                    return e.first == key;
                });
    }

    /// \brief Insert a key-value pair
    void insert(const key_type &key, const value_type &value) {

        auto it = this->find(key);

        if (it == this->end()) {
            _data.emplace_back(key, value);
        }
        else {
            it->second = value;
        }
    }

    /// \brief Get a value by index
    value_type& at(const key_type &key) {
        auto it = this->find(key);

        if (it == this->end()) {
            throw std::out_of_range("record name does not exist");
        }
        return it->second;
    }

    /// \brief Get the map size
    size_t size() const {
        return _data.size();
    }

    /// \brief Clear data
    void clear() { _data.clear(); }
};


#endif  // TINY_RECORD_H_
