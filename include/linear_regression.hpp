#ifndef LINEAR_REGRESSION_HPP
#define LINEAR_REGRESSION_HPP

#include <Eigen/Dense>
#include <string>

namespace lr {

    enum class Solver { Normal, Ridge, GD };

    class LinearRegression {
      private:
        // learned params
        Eigen::VectorXd theta_; // weights (d,)
        double intercept_;      // bias term

        // config
        Solver solver_;      // solver type
        double l2_;          // l2 regularization strength
        double lr_;          // learning rate for gd
        int epochs_;         // iterations for gd
        bool fit_intercept_; // add bias column and learn intercept
        double tol_;         // early stopping tolerance for gd

        // state
        bool fitted_; // model fitted flag

        // internal helpers
        Eigen::VectorXd normal_eq_fit(const Eigen::MatrixXd& Xb,
            const Eigen::VectorXd& y) const; // closed form

        Eigen::VectorXd ridge_fit(const Eigen::MatrixXd& Xb,
            const Eigen::VectorXd& y) const; // closed form + l2

        Eigen::VectorXd gd_fit(const Eigen::MatrixXd& Xb,
            const Eigen::VectorXd& y) const; // full-batch gd

        Eigen::MatrixXd add_bias_column(
            const Eigen::MatrixXd& X) const; // [X | 1]

        void split_theta_with_bias(
            const Eigen::VectorXd& w_full); // set theta_, intercept_

      public:
        // ctor-only config
        LinearRegression(Solver solver = Solver::Normal,
            double l2 = 0.0,
            double lr = 0.01,
            int epochs = 1000,
            bool fit_intercept = true,
            double tol = 1e-6);

        // fit on X(n,d), y(n)
        void fit(const Eigen::MatrixXd& X, const Eigen::VectorXd& y);

        // predict yhat(n) for X(n,d)
        Eigen::VectorXd predict(const Eigen::MatrixXd& X) const;

        // accessors
        Eigen::VectorXd coefficients() const; // returns theta_
        double intercept() const;             // returns bias
        bool is_fitted() const {
            return fitted_;
        }
    };

} // namespace lr

#endif
