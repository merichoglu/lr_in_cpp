/*
 * dataset.hpp
 */

#ifndef DATASET_HPP
#define DATASET_HPP

#include <Eigen/Dense>
#include <string>

namespace lr {
    // load csv -> last column is target
    void load_csv(const std::string& file_name,
        Eigen::MatrixXd& X,
        Eigen::VectorXd& y,
        bool has_header);

    // random split into train/test
    void train_test_split(const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y,
        double test_ratio,
        Eigen::MatrixXd& X_train,
        Eigen::VectorXd& y_train,
        Eigen::MatrixXd& X_test,
        Eigen::VectorXd& y_test);

} // namespace lr

#endif
