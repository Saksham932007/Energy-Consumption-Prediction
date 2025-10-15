# Energy Consumption Prediction using LSTM Neural Networks

This project demonstrates the use of a Long Short-Term Memory (LSTM) neural network to forecast energy consumption based on historical time-series data. The model is built with TensorFlow and Keras and is designed to run in a Google Colab environment. It includes data preprocessing, model training, evaluation, and visualization of the results.

![Project Screenshot](images/results_plot.png)
*Note: You will need to take a screenshot of your final plot and save it as `results_plot.png` in an `images` folder within your project.*

---

## 📋 Overview

Predicting energy consumption is a critical task for power companies, grid operators, and consumers to optimize resource allocation, manage costs, and ensure grid stability. This project tackles this forecasting problem by leveraging an LSTM network, a type of recurrent neural network (RNN) well-suited for learning patterns from sequential data.

The notebook is structured step-by-step to provide a clear and educational workflow.

---

## ✨ Key Features

-   **Time-Series Forecasting**: Predicts the next hour's energy consumption based on the previous 24 hours of data.
-   **Deep Learning Model**: Implements a stacked LSTM model with Dropout layers to capture complex temporal patterns and prevent overfitting.
-   **Data Preprocessing**: Includes data scaling using `MinMaxScaler` to prepare the data for the neural network.
-   **Synthetic Data Generator**: Comes with a built-in function to generate a realistic, five-year hourly energy consumption dataset, making the project runnable out-of-the-box.
-   **Performance Evaluation**: Calculates key regression metrics, including R-squared (R²), Mean Absolute Error (MAE), Root Mean Squared Error (RMSE), and Mean Absolute Percentage Error (MAPE).
-   **Visualization**: Plots the predicted values against the actual values for a clear visual assessment of the model's performance.

---

## 🛠️ Tech Stack

-   **Python 3**
-   **Google Colab**
-   **TensorFlow & Keras**: For building and training the LSTM neural network.
-   **Pandas**: For data manipulation and loading.
-   **NumPy**: For numerical operations.
-   **Scikit-learn**: For data preprocessing (scaling) and splitting.
-   **Matplotlib**: For plotting and visualizing the results.

---

## 🚀 Getting Started

The easiest way to run this project is by using Google Colab.

### Prerequisites

-   A Google Account to access Google Colab.

### ⚙️ Installation & Usage

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/energy-consumption-prediction.git](https://github.com/your-username/energy-consumption-prediction.git)
    cd energy-consumption-prediction
    ```

2.  **Upload to Google Colab:**
    -   Go to [https://colab.research.google.com/](https://colab.research.google.com/).
    -   Click on `File` > `Upload notebook...` and select the `Untitled16.ipynb` file from the cloned repository.

3.  **Run the Notebook Cells:**
    -   The notebook is divided into sequential cells. Run each cell from top to bottom.
    -   The first cell will automatically install all the necessary libraries.
    ```python
    !pip install tensorflow pandas scikit-learn matplotlib -q
    ```

### 🔌 Using Your Own Data

The notebook is configured to work with your custom time-series data.

1.  **Prepare your CSV file**. It should have at least two columns: one for the timestamp and one for the energy consumption value.
2.  **Upload the CSV** to your Google Colab session.
3.  In **Cell 3** of the notebook, comment out the synthetic data generation part and uncomment the `pd.read_csv` line:

    ```python
    # Comment this line out:
    # df = generate_synthetic_data()

    # Uncomment this line and change the filename to match yours:
    df = pd.read_csv('your_data.csv', index_col='timestamp', parse_dates=True)
    ```
4.  Run the rest of the notebook cells as usual.

---

## 📈 Results & Evaluation

The model was trained on the synthetic dataset and evaluated on a 20% hold-out test set. The results demonstrate the model's high accuracy in forecasting energy consumption.

| Metric                        | Value       |
| ----------------------------- | ----------- |
| **R-squared (R²)** | 0.9578      |
| **Prediction Accuracy (100 - MAPE)** | **91.75%** |
| **Mean Absolute Error (MAE)** | 13.39 kWh   |
| **RMSE** | 16.95 kWh   |

The **R-squared value of ~0.96** indicates that the model can explain 96% of the variance in the energy consumption data, which is an excellent fit. The final plot provides a visual confirmation, showing the predicted values closely tracking the actual values.

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome! Feel free to check the [issues page](https://github.com/your-username/energy-consumption-prediction/issues).

---

## 📄 License

This project is licensed under the MIT License. See the `LICENSE` file for more details.
