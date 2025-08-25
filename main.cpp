#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include <iomanip>
#include <cmath>
#include <Eigen/Dense>
#include "linear_regression.hpp"
#include "dataset.hpp"
#include "utils.hpp"

void demo_synthetic_data() {
    std::cout << "\n=== Synthetic Data Demo ===\n";
    
    // Generate synthetic dataset: y = 3*x1 + 2*x2 - 1*x3 + 5 + noise
    const int n_samples = 1000;
    const int n_features = 3;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> noise(0.0, 0.5);
    std::uniform_real_distribution<> uniform(-2.0, 2.0);
    
    Eigen::MatrixXd X(n_samples, n_features);
    Eigen::VectorXd y(n_samples);
    Eigen::VectorXd true_coeff(n_features);
    true_coeff << 3.0, 2.0, -1.0;
    double true_intercept = 5.0;
    
    // Generate data
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = uniform(gen);
        }
        y(i) = X.row(i).dot(true_coeff) + true_intercept + noise(gen);
    }
    
    std::cout << "True coefficients: " << true_coeff.transpose() << std::endl;
    std::cout << "True intercept: " << true_intercept << std::endl;
    
    // Split data (80% train, 20% test)
    Eigen::MatrixXd X_train, X_test;
    Eigen::VectorXd y_train, y_test;
    lr::train_test_split(X, y, 0.2, X_train, y_train, X_test, y_test);
    
    // Test different solvers
    std::vector<lr::Solver> solvers = {lr::Solver::Normal, lr::Solver::Ridge, lr::Solver::GD};
    std::vector<std::string> solver_names = {"Normal Equation", "Ridge", "Gradient Descent"};
    
    for (size_t i = 0; i < solvers.size(); ++i) {
        std::cout << "\n--- " << solver_names[i] << " ---\n";
        
        lr::LinearRegression model(solvers[i], 0.01, 0.01, 1000, true, 1e-6);
        
        lr::Timer timer;
        model.fit(X_train, y_train);
        double elapsed = timer.elapsed_seconds();
        
        Eigen::VectorXd y_pred = model.predict(X_test);
        
        // Calculate metrics
        double test_mse = lr::mse(y_test, y_pred);
        double test_rmse = std::sqrt(test_mse);
        double test_r2 = lr::r2_score(y_test, y_pred);
        
        std::cout << "Learned coefficients: " << model.coefficients().transpose() << std::endl;
        std::cout << "Learned intercept: " << model.intercept() << std::endl;
        std::cout << "Training time: " << std::fixed << std::setprecision(4) << elapsed << " s" << std::endl;
        std::cout << "Test RMSE: " << test_rmse << std::endl;
        std::cout << "Test R²: " << test_r2 << std::endl;
    }
}

void demo_regularization() {
    std::cout << "\n=== Regularization Demo ===\n";
    
    // Generate data with more features than samples (overfitting scenario)
    const int n_samples = 50;
    const int n_features = 20;
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal(0.0, 1.0);
    
    Eigen::MatrixXd X(n_samples, n_features);
    Eigen::VectorXd y(n_samples);
    
    for (int i = 0; i < n_samples; ++i) {
        for (int j = 0; j < n_features; ++j) {
            X(i, j) = normal(gen);
        }
        // Only first 3 features are actually relevant
        y(i) = X(i, 0) + 2*X(i, 1) - X(i, 2) + 0.1*normal(gen);
    }
    
    std::vector<double> l2_values = {0.0, 0.01, 0.1, 1.0, 10.0};
    
    std::cout << "L2 Regularization Effect:\n";
    std::cout << "L2\tCoeff Norm\tFirst 3 Coeffs\n";
    
    for (double l2 : l2_values) {
        lr::LinearRegression model(lr::Solver::Ridge, l2, 0.01, 1000, true, 1e-6);
        model.fit(X, y);
        
        Eigen::VectorXd coeffs = model.coefficients();
        double coeff_norm = coeffs.norm();
        
        std::cout << std::fixed << std::setprecision(3) << l2 << "\t" 
                  << coeff_norm << "\t\t[" 
                  << coeffs(0) << ", " << coeffs(1) << ", " << coeffs(2) << "]\n";
    }
}

void load_and_demo_csv(const std::string& filename) {
    std::cout << "\n=== CSV Dataset Demo: " << filename << " ===\n";
    
    try {
        // Load CSV data
        Eigen::MatrixXd X, X_train, X_test;
        Eigen::VectorXd y, y_train, y_test;
        
        lr::load_csv(filename, X, y, true); // has_header = true
        
        std::cout << "Dataset loaded successfully!\n";
        std::cout << "Samples: " << X.rows() << ", Features: " << X.cols() << std::endl;
        std::cout << "Target range: [" << y.minCoeff() << ", " << y.maxCoeff() << "]\n";
        std::cout << "Target mean: " << y.mean() << std::endl;
        
        // Split into train/test (80/20)
        lr::train_test_split(X, y, 0.2, X_train, y_train, X_test, y_test);
        std::cout << "Train size: " << X_train.rows() << ", Test size: " << X_test.rows() << std::endl;
        
        // Test all solvers
        std::vector<lr::Solver> solvers = {lr::Solver::Normal, lr::Solver::Ridge, lr::Solver::GD};
        std::vector<std::string> solver_names = {"Normal Equation", "Ridge", "Gradient Descent"};
        
        for (size_t i = 0; i < solvers.size(); ++i) {
            std::cout << "\n--- " << solver_names[i] << " ---\n";
            
            // Adjust parameters based on solver
            double l2_reg = (solvers[i] == lr::Solver::Ridge) ? 0.1 : 0.0;
            lr::LinearRegression model(solvers[i], l2_reg, 0.01, 5000, true, 1e-8);
            
            lr::Timer timer;
            model.fit(X_train, y_train);
            double elapsed = timer.elapsed_seconds();
            
            // Make predictions
            Eigen::VectorXd y_train_pred = model.predict(X_train);
            Eigen::VectorXd y_test_pred = model.predict(X_test);
            
            // Calculate metrics
            double train_mse = lr::mse(y_train, y_train_pred);
            double test_mse = lr::mse(y_test, y_test_pred);
            double train_rmse = std::sqrt(train_mse);
            double test_rmse = std::sqrt(test_mse);
            double train_r2 = lr::r2_score(y_train, y_train_pred);
            double test_r2 = lr::r2_score(y_test, y_test_pred);
            
            std::cout << "Training time: " << std::fixed << std::setprecision(4) << elapsed << " s" << std::endl;
            std::cout << "Coefficients: " << model.coefficients().transpose() << std::endl;
            std::cout << "Intercept: " << model.intercept() << std::endl;
            std::cout << "Train RMSE: " << train_rmse << ", R²: " << train_r2 << std::endl;
            std::cout << "Test RMSE: " << test_rmse << ", R²: " << test_r2 << std::endl;
            
            // Check for overfitting
            if (test_rmse > train_rmse * 1.2) {
                std::cout << "⚠️  Potential overfitting detected!" << std::endl;
            }
        }
        
    } catch (const std::exception& e) {
        std::cout << "Error loading CSV: " << e.what() << std::endl;
    }
}

int main() {
    std::cout << "Linear Regression from Scratch - Demo\n";
    std::cout << "=====================================\n";
    
    try {
        // Demo 1: Synthetic data with known ground truth
        demo_synthetic_data();
        
        // Demo 2: Regularization effects
        demo_regularization();
        
        // Demo 3: Load real dataset (wine quality)
        load_and_demo_csv("data/WineQT.csv");
        
        std::cout << "\n=== Demo Complete ===\n";
        
    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
    
    return 0;
}