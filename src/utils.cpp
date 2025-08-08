/*
 * utils.cpp
 */

#include "utils.hpp"

#include <cmath>
#include <stdexcept>

namespace lr {

    double mse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("mse: size mismatch");
        }
        const Eigen::VectorXd diff = y_true - y_pred;
        return diff.squaredNorm() / static_cast<double>(y_true.size());
    }

    double r2_score(const Eigen::VectorXd& y_true,
        const Eigen::VectorXd& y_pred) {
        if (y_true.size() != y_pred.size()) {
            throw std::invalid_argument("r2_score: size mismatch");
        }

        const double mean_y = y_true.mean();
        const double ss_res = (y_true - y_pred).squaredNorm();
        const double ss_tot = (y_true.array() - mean_y).square().sum();

        // handle constant target: define r2 as 1 if perfect, else 0
        if (ss_tot == 0.0) {
            return (ss_res == 0.0) ? 1.0 : 0.0;
        }
        return 1.0 - (ss_res / ss_tot);
    }

    // start clock at construction
    Timer::Timer() : start_time_(std::chrono::steady_clock::now()) {
    }

    // seconds since construction
    double Timer::elapsed_seconds() const {
        const auto now = std::chrono::steady_clock::now();
        return std::chrono::duration<double>(now - start_time_).count();
    }

} // namespace lr
