/*
 * dataset.cpp
 */

#include "dataset.hpp"

#include <algorithm>
#include <fstream>
#include <numeric>
#include <random>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace lr {
    void load_csv(const std::string& file_name,
        Eigen::MatrixXd& X,
        Eigen::VectorXd& y,
        bool has_header) {
        std::ifstream file(file_name);
        if (!file.is_open()) {
            throw std::runtime_error("could not open file: " + file_name);
        }

        std::string line;

        // skip header if present
        if (has_header && std::getline(file, line)) {
            // no-op
        }

        std::vector<std::vector<double>> rows;
        rows.reserve(1024); // tiny pre-reserve

        std::size_t expected_cols = 0;

        while (std::getline(file, line)) {
            // skip empty lines
            if (line.empty())
                continue;

            std::stringstream ss(line);
            std::string cell;
            std::vector<double> row;
            row.reserve(8);

            while (std::getline(ss, cell, ',')) {
                // skip purely empty trailing cells (basic guard against
                // trailing comma)
                if (cell.empty()) {
                    continue;
                }
                row.push_back(std::stod(cell)); // throws on bad numeric -> good
            }

            if (row.empty()) {
                continue; // line had no usable values
            }

            if (expected_cols == 0) {
                expected_cols = row.size();
            } else if (row.size() != expected_cols) {
                throw std::runtime_error(
                    "inconsistent column count in csv: got " +
                    std::to_string(row.size()) + " vs expected " +
                    std::to_string(expected_cols));
            }

            rows.push_back(std::move(row));
        }

        if (rows.empty()) {
            throw std::runtime_error("csv has no data rows: " + file_name);
        }
        if (expected_cols < 2) {
            throw std::runtime_error(
                "csv must have at least 2 columns (features + target)");
        }

        const Eigen::Index n_rows = static_cast<Eigen::Index>(rows.size());
        const Eigen::Index n_cols = static_cast<Eigen::Index>(expected_cols);

        X.resize(n_rows, n_cols - 1);
        y.resize(n_rows);

        for (Eigen::Index i = 0; i < n_rows; ++i) {
            // fill features
            for (Eigen::Index j = 0; j < n_cols - 1; ++j) {
                X(i, j) = rows[static_cast<std::size_t>(i)]
                              [static_cast<std::size_t>(j)];
            }
            // last column is target
            y(i) = rows[static_cast<std::size_t>(i)]
                       [static_cast<std::size_t>(n_cols - 1)];
        }
    }

    void train_test_split(const Eigen::MatrixXd& X,
        const Eigen::VectorXd& y,
        double test_ratio,
        Eigen::MatrixXd& X_train,
        Eigen::VectorXd& y_train,
        Eigen::MatrixXd& X_test,
        Eigen::VectorXd& y_test) {
        // basic checks
        if (X.rows() != y.size()) {
            throw std::runtime_error("X.rows() must equal y.size()");
        }
        if (X.rows() == 0) {
            throw std::runtime_error("cannot split empty dataset");
        }
        if (!(test_ratio >= 0.0 && test_ratio <= 1.0)) {
            throw std::runtime_error("test_ratio must be in [0, 1]");
        }

        const Eigen::Index n = X.rows();
        std::vector<Eigen::Index> idx(static_cast<std::size_t>(n));
        std::iota(idx.begin(), idx.end(), 0);

        // shuffle indices
        std::random_device rd;
        std::mt19937 gen(rd());
        std::shuffle(idx.begin(), idx.end(), gen);

        const Eigen::Index test_size =
            static_cast<Eigen::Index>(n * test_ratio);
        const Eigen::Index train_size = n - test_size;

        X_train.resize(train_size, X.cols());
        y_train.resize(train_size);
        X_test.resize(test_size, X.cols());
        y_test.resize(test_size);

        // fill train
        for (Eigen::Index i = 0; i < train_size; ++i) {
            const Eigen::Index k = idx[static_cast<std::size_t>(i)];
            X_train.row(i) = X.row(k);
            y_train(i) = y(k);
        }
        // fill test
        for (Eigen::Index i = 0; i < test_size; ++i) {
            const Eigen::Index k =
                idx[static_cast<std::size_t>(train_size + i)];
            X_test.row(i) = X.row(k);
            y_test(i) = y(k);
        }
    }

} // namespace lr
