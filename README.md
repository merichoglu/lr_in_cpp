# Linear Regression from Scratch (C++)

This project implements **Linear Regression** using three different solvers:

* **Normal Equation** (closed-form)
* **Ridge Regression** (L2 regularization)
* **Gradient Descent** (iterative optimization)

The code leverages the **Eigen** library for efficient linear algebra operations and includes a small demo using both **synthetic data** and a real-world CSV dataset, `WineQT.csv`.

---

## 📁 Project Structure

```bash
linear-regression-cpp/
├── build/
├── data/
│   ├── sample.csv
│   └── WineQT.csv
├── include/
│   ├── dataset.hpp
│   ├── linear_regression.hpp
│   └── utils.hpp
├── src/
│   ├── dataset.cpp
│   ├── linear_regression.cpp
│   └── utils.cpp
├── tests/
│   └── test_linear_regression.cpp
├── main.cpp
├── CMakeLists.txt
└── README.md
```

---

## Build & Run

To build and run the demo, follow these steps in your terminal:

\`\`\`bash
mkdir build && cd build
cmake ..
make
./linear_regression
\`\`\`

---

## Results

### Synthetic Data Demo

The model successfully recovered the true coefficients from the synthetic data. The results show that all three solvers produce very similar, accurate outputs.

**True coefficients:** `[3, 2, -1]`, intercept `5`

| Solver            | Coefficients (approx)       | Intercept | Test RMSE | Test R² |
|-------------------|-----------------------------|-----------|-----------|---------|
| **Normal Equation** | `[2.984, 2.008, -1.003]`     | `4.988`     | `0.4675`    | `0.9876`  |
| **Ridge Regression** | `[2.984, 2.008, -1.003]`     | `4.988`     | `0.4675`    | `0.9876`  |
| **Gradient Descent** | `[2.962, 1.993, -0.995]`     | `4.987`     | `0.4726`    | `0.9873`  |

---

### Regularization Demo

Increasing the L2 penalty correctly shrinks the coefficient norms, as expected in Ridge Regression.

| L2 Penalty | Norm  | First 3 Coeffs (approx)   |
|------------|-------|---------------------------|
| **0.0** | `2.460` | `[1.002, 2.002, -1.015]`    |
| **0.1** | `2.447` | `[0.995, 1.991, -1.014]`    |
| **1.0** | `2.341` | `[0.933, 1.894, -1.002]`    |
| **10.0** | `1.785` | `[0.617, 1.366, -0.875]`    |

---

### WineQT Dataset

This real-world dataset highlights the limitations of a simple linear model.

* **Samples:** **1143**
* **Features:** **12**
* **Train/Test Split:** **915 / 228**

| Solver            | Train RMSE | Test RMSE | Test R² |
|-------------------|------------|-----------|---------|
| **Normal Equation** | `415.35`     | `410.12`    | `0.2002`  |
| **Ridge Regression** | `415.35`     | `410.12`    | `0.2003`  |
| **Gradient Descent** | `415.36`     | `409.95`    | `0.2009`  |

**Note:** The low R² value of approximately `0.20` indicates that the dataset is noisy and not well-suited for a linear regression model.
