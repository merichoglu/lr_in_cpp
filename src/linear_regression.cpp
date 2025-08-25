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
        if (solver_ == Solver::GD && !(lr_ > 0.0)) {
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
        - implement split_theta_with_bias(): separate weights and bias from
       w_full
        - implement predict(): Xθ + b
        - implement coefficients(): return θ
        - implement intercept(): return b
    */

    Eigen::VectorXd LinearRegression::normal_eq_fit(const Eigen::MatrixXd& Xb,
        const Eigen::VectorXd& y) const {
        // Closed-form solution: (XᵀX)⁻¹Xᵀy
        Eigen::MatrixXd XtX = Xb.transpose() * Xb;
        Eigen::VectorXd XtY = Xb.transpose() * y;
        return XtX.ldlt().solve(XtY);
    }

    Eigen::VectorXd LinearRegression::ridge_fit(const Eigen::MatrixXd& Xb,
        const Eigen::VectorXd& y) const {
        // Closed-form solution with L2 regularization: (XᵀX + λI)⁻¹Xᵀy
        int d = Xb.cols();
        Eigen::MatrixXd I = Eigen::MatrixXd::Identity(d, d);
        if (fit_intercept_) {
            I(d - 1, d - 1) = 0.0; // no reg on bias term
        }
        Eigen::MatrixXd XtX = Xb.transpose() * Xb;
        Eigen::VectorXd XtY = Xb.transpose() * y;
        return (XtX + l2_ * I).ldlt().solve(XtY);
    }

    Eigen::VectorXd LinearRegression::gd_fit(const Eigen::MatrixXd& Xb,
        const Eigen::VectorXd& y) const {
        int n = Xb.rows();
        int d = Xb.cols();
        Eigen::VectorXd w = Eigen::VectorXd::Zero(d);

        for (int epoch = 0; epoch < epochs_; ++epoch) {
            Eigen::VectorXd y_pred = Xb * w;
            Eigen::VectorXd error = y_pred - y;
            Eigen::VectorXd grad = (Xb.transpose() * error) / n;

            // add l2 reg (skip bias term if enabled)
            if (l2_ > 0.0) {
                Eigen::VectorXd w_reg = w;
                if (fit_intercept_) {
                    w_reg(d - 1) = 0.0;
                }
                grad += l2_ * w_reg;
            }

            Eigen::VectorXd w_new = w - lr_ * grad;

            // monitor divergence: if weights blow up, stop
            if (!w_new.allFinite()) {
                throw std::runtime_error("Divergence detected: try lowering "
                                         "learning rate or scaling features.");
            }

            // early stopping
            if ((w_new - w).norm() < tol_) {
                w = w_new;
                break;
            }

            w = w_new;
        }

        return w;
    }

    Eigen::MatrixXd LinearRegression::add_bias_column(
        const Eigen::MatrixXd& X) const {
        // Append a column of ones to X for the bias term
        Eigen::MatrixXd Xb(X.rows(), X.cols() + 1);
        Xb << X, Eigen::VectorXd::Ones(X.rows());
        return Xb;
    }

    void LinearRegression::split_theta_with_bias(
        const Eigen::VectorXd& w_full) {
        // Separate weights and bias from w_full
        int d = w_full.size();
        theta_ = w_full.head(d - 1);
        intercept_ = w_full(d - 1);
    }

    Eigen::VectorXd LinearRegression::predict(const Eigen::MatrixXd& X) const {
        if (!fitted_) {
            throw std::runtime_error("Model is not fitted yet");
        }
        if (X.cols() != theta_.size()) {
            throw std::invalid_argument(
                "X.cols() must match number of features");
        }
        return X * theta_ + Eigen::VectorXd::Constant(X.rows(), intercept_);
    }

    Eigen::VectorXd LinearRegression::coefficients() const {
        if (!fitted_) {
            throw std::runtime_error("Model is not fitted yet");
        }
        return theta_;
    }

    double LinearRegression::intercept() const {
        if (!fitted_) {
            throw std::runtime_error("Model is not fitted yet");
        }
        return intercept_;
    }
} // namespace lr
