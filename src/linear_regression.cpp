#include "linear_regression.hpp"
#include <stdexcept>
#include <cmath>

namespace lr {
    LinearRegression::LinearRegression(Solver solver,
        double l2,
        double lr,
        int epochs,
        bool fit_intercept,
        double tol) :
        solver_(solver),
        l2_(l2),
        lr_(lr),
        epochs_(epochs),
        fit_intercept_(fit_intercept),
        tol_(tol),
        fitted_(false) {
        if (epochs_ <= 0) {
            throw std::invalid_argument("epochs must be > 0");
        }
        if (!(lr_ > 0.0 && solver_ == Solver::GD)) {
            throw std::invalid_argument("GD requires lr > 0.0");
        }
        if (!(tol > 0.0)) {
            throw std::invalid_argument("tol must be > 0.0");
        }
        if (!(l2_ >= 0.0)) {
            throw std::invalid_argument("l2 must be >= 0.0");
        }
    }

    void LinearRegression::fit(const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y) {
        if (X.rows() != y.size()) {
            throw std::invalid_argument("X.rows() must match y.size()");
        }
        if (X.cols() == 0) {
            throw std::invalid_argument("X must have at least one feature");
        }

        Eigen::MatrixXd Xb = fit_intercept_ ? add_bias_column(X) : X;

        Eigen::VectorXd w_full;
        switch (solver_) {
        case Solver::Normal: w_full = normal_eq_fit(Xb, y); break;
        case Solver::Ridge: w_full = ridge_fit(Xb, y); break;
        case Solver::GD: w_full = gd_fit(Xb, y); break;
        default: throw std::invalid_argument("Unknown solver type");
        }

        if (fit_intercept_) {
            split_theta_with_bias(w_full);
        } else {
            theta_ = w_full;
            intercept_ = 0.0; // no bias term
        }
        fitted_ = true;
    }

    /*
        TODOS:
        - implement normal_eq_fit(): closed-form (XᵀX)⁻¹Xᵀy
        - implement ridge_fit(): closed-form (XᵀX + λI)⁻¹Xᵀy, skip bias reg
        - implement gd_fit(): full-batch gradient descent with optional L2
        - implement add_bias_column(): append col of 1s to X
        - implement split_theta_with_bias(): separate weights and bias from w_full
        - implement predict(): Xθ + b
        - implement coefficients(): return θ
        - implement intercept(): return b
    */

} // namespace lr