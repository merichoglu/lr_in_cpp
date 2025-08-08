/*
 * utils.hpp
 */

#ifndef UTILS_HPP
#define UTILS_HPP

#include <Eigen/Dense>
#include <chrono>

namespace lr {

    // mse between y_true and y_pred
    double mse(const Eigen::VectorXd& y_true, const Eigen::VectorXd& y_pred);

    // r^2 with constant-target guard
    double r2_score(const Eigen::VectorXd& y_true,
        const Eigen::VectorXd& y_pred);

    // simple wall-clock timer
    class Timer {
      private:
        std::chrono::steady_clock::time_point start_time_;

      public:
        Timer();
        double elapsed_seconds() const;
    };

} // namespace lr

#endif
