import numpy as np
import matplotlib.pyplot as plt
from lab9.ex1234 import generate_time_series, train_test_split

def train_ar(y_train: np.ndarray, p) -> np.ndarray:
    train_len = len(y_train)

    Y_train = np.zeros((train_len - p, p))
    y_target_train = y_train[p:]
    for i in range(train_len - p):
        Y_train[i, :] = y_train[i: i + p][::-1]
    ar_coefficients = np.linalg.lstsq(Y_train, y_target_train)[0]
    return ar_coefficients

def train_ar_with_lags(y_train: np.ndarray, lags: np.ndarray) -> np.ndarray:
    p = len(lags)
    max_lag = lags.max()
    train_len = len(y_train)
    Y_train = np.zeros((train_len - max_lag, p))
    y_target_train = y_train[max_lag:]
    for i in range(train_len - max_lag):
        Y_train[i, :] = y_train[i + lags-1][::-1]
    ar_coefficients = np.linalg.lstsq(Y_train, y_target_train)[0]
    return ar_coefficients

def test_ar(y_train: np.ndarray, y_test: np.ndarray, ar_coefficients: np.ndarray) -> np.ndarray:
    test_len = len(y_test)
    p = len(ar_coefficients)
    Y_test = np.zeros((test_len, p))
    test_history = np.concatenate((y_train[-p:], y_test))
    for i in range(test_len):
        Y_test[i, :] = test_history[i: i + p][::-1]
    y_pred = Y_test @ ar_coefficients
    return y_pred

def test_ar_with_lags(y_train: np.ndarray, y_test: np.ndarray,
                      ar_coefficients: np.ndarray, lags: np.ndarray) -> np.ndarray:
    test_len = len(y_test)
    p = len(ar_coefficients)
    max_lag = lags.max()
    Y_test = np.zeros((test_len, p))
    test_history = np.concatenate((y_train[-max_lag:], y_test))
    for i in range(test_len):
        Y_test[i, :] = test_history[i + lags-1][::-1]
    y_pred = Y_test @ ar_coefficients
    return y_pred

def greedy_select_lag(y_train: np.ndarray, lags: list[int], P:int) -> int:
    p = len(lags)
    lags_set = set(lags)
    max_lag = max(lags_set) if p > 0 else 0
    min_loss = float('inf')
    best_lag = 0
    for lag in range(1, P+1):
        if lag not in lags_set:
            lags.append(lag)
            lags_array = np.array(lags)
            coef = train_ar_with_lags(y_train, lags_array)
            lags.pop()
            train_hist = y_train[:max(max_lag, lag)]
            train_test_arr = y_train[max(max_lag, lag):]
            y_pred = test_ar_with_lags(train_hist, train_test_arr, coef, lags_array)
            cur_loss = rmse_loss(train_test_arr, y_pred)
            if cur_loss < min_loss:
                min_loss = cur_loss
                best_lag = lag
    return best_lag

def rmse_loss(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    return np.sqrt(1/len(y_pred)*np.sum((y_true-y_pred)**2))

def plot_preds(y_true: np.ndarray, y_pred: np.ndarray) -> None:
    plt.plot(y_true, label='True')
    plt.plot(y_pred, label='Predicted')
    plt.legend()
    plt.savefig('ex123_greedy_lags.pdf')
    plt.show()

def run_cur():
    t_s = generate_time_series(N=1000)
    train_s, test_s = train_test_split(t_s)
    p = 10
    model_coef = train_ar(train_s, p=p)
    print('Train coefficients:', model_coef)
    y_pred = test_ar(train_s, test_s, model_coef)
    test_loss = rmse_loss(test_s, y_pred)
    print(f'p={p}, Test RMSE: {test_loss}')
    #plot_preds(test_s, y_pred)
    small_p = 5
    init_lags = []
    for i in range(small_p):
        top_lag = greedy_select_lag(train_s, init_lags, p)
        init_lags.append(top_lag)
    print(f'init_lags={init_lags}')
    model_coef = train_ar_with_lags(train_s, np.array(init_lags))
    print(f'model_coef with lags={model_coef}')
    y_pred = test_ar_with_lags(train_s, test_s, model_coef, np.array(init_lags))
    loss = rmse_loss(test_s, y_pred)
    print(f'RMSE with greedy lags: {loss}')
    plot_preds(test_s, y_pred)


if __name__ == "__main__":
    run_cur()
