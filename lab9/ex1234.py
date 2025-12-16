import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from statsmodels.tsa.arima.model import ARIMA
import warnings
warnings.filterwarnings('ignore')

def my_plot_series(series, titles):
    num_rows = len(series)
    fig, ax = plt.subplots(nrows=num_rows, ncols=1)
    for i, s in enumerate(series):
        ax[i].plot(np.arange(len(s)), s)
        ax[i].set_title(titles[i])
    plt.tight_layout()
    plt.show()

def train_test_split(data, train_size=0.85, train_percent=1.0):
    N = len(data)
    train_end = int(N * train_size)
    train_start = train_end - int(train_percent * train_end)
    # train_len = train_end - train_start
    # test_len = N - train_end
    # print(train_start, train_end, train_len, test_len)
    y_train = data[train_start:train_end]
    y_test = data[train_end:]
    return y_train, y_test

def train_and_predict_ma(y_train, y_test, q=10):
    train_len = len(y_train)
    Y_train = np.zeros((train_len - q, q))
    mu = np.mean(y_train)
    std = np.std(y_train)
    error_norm = stats.norm(loc=mu, scale=np.sqrt(std))
    for i in range(train_len - q):
        Y_train[i, :] = error_norm.rvs(size=q)
    y_target_train = y_train[q:] - mu
    ma_coefficients = np.linalg.lstsq(Y_train, y_target_train)[0]
    test_len = len(y_test)
    Y_test = np.zeros((test_len, q))
    test_history = np.concatenate((y_train[-q:], y_test))
    for i in range(test_len):
        Y_test[i, :] = error_norm.rvs(size=q)

    y_pred = Y_test @ ma_coefficients + mu
    return y_pred

def fit(train_series, q=10, p=0, i=0):
    model = ARIMA(endog=train_series, order=(p, i, q))
    model_fit = model.fit()
    #y_pred = model_fit.forecast(steps=len(test_series))
    # my_plot_series([time_series, smoothed,
    #                 test_series, y_pred],
    #                ['Original', 'Smoothed', 'Test', 'Prediction'])
    return model_fit

def run_code():
    #time_diff = smoothed - trend_component
    train_series, test_series = train_test_split(smoothed)
    i_diff = 2
    best_params = (2, i_diff, 3)
    best_model = None
    max_p, max_q = (10, 10)
    for p in range(9, max_p):
        for q in range(9, max_q):
            try:
                model_fit = fit(train_series, p=p, i=i_diff, q=q)
            except np.linalg.LinAlgError:
                continue
            if best_model is None or model_fit.aic < best_model.aic:
                best_model = model_fit
                best_params = (p, i_diff, q)
    print(best_params)

    y_pred_train = best_model.predict(start=0, end=len(train_series)-1)
    rmse_train = np.sqrt(1 / len(y_pred_train) *
                         np.sum((time_series[:len(train_series)] - y_pred_train) ** 2))
    y_pred_test = best_model.forecast(steps=len(test_series))
    rmse_test = np.sqrt(1 / len(y_pred_test) *
                        np.sum((time_series[len(train_series):] - y_pred_test) ** 2))
    print("Rmse train:", rmse_train)
    print("Rmse test:", rmse_test)

    fig, ax = plt.subplots(nrows=1, ncols=2)
    ax[0].plot(time_series[:len(train_series)], label='train')
    ax[0].plot(y_pred_train, label='y_pred')
    ax[1].plot(time_series[len(train_series):], label='test')
    ax[1].plot(y_pred_test, label='y_pred')
    ax[0].legend()
    ax[1].legend()
    #plt.savefig('Train_and_Test_Predictions.pdf')
    plt.show()

def generate_time_series(N: int) -> np.ndarray:
    np.random.seed(77)
    polynom2 = np.array([0.05, -1, 7000])
    t = np.arange(N)
    trend_component = np.column_stack([t ** 2, t, np.ones_like(t)]) @ polynom2
    f1, f2 = 7 / N, 30 / N
    A1, A2 = 2 * N, 5 * N
    seasonal_component = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.cos(2 * np.pi * f2 * t)
    mu, sigma = 1500, N // 2
    noise_component = np.random.normal(mu, sigma, size=N)
    time_series = trend_component + seasonal_component + noise_component
    return time_series

if __name__ == '__main__':
    N = 1000
    time_series = generate_time_series(N)
    alpha = 0.7
    smoothed = np.zeros_like(time_series)
    smoothed[0] = time_series[0]
    for i in range(1, N):
        smoothed[i] = alpha * time_series[i] + (1 - alpha) * smoothed[i - 1]
    run_code()


