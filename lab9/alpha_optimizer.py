import numpy as np
import matplotlib.pyplot as plt
from ex1234 import train_test_split, fit, generate_time_series

def theta_from_alpha(alpha):
    return np.tan(np.pi * (alpha - 0.5))

def alpha_from_theta(theta):
    return np.arctan(theta) / np.pi + 0.5

def dalpha_dtheta(theta):
    return 1.0 / (np.pi * (1 + theta)**2)

def calc_alpha_loss_gradient(alpha: float, x:np.ndarray, smoothed: np.ndarray) -> float:
    N = len(smoothed)
    gradient = np.zeros_like(smoothed)
    gradient[0] = 0

    for i in range(1, N):
        gradient[i] = x[i] - smoothed[i - 1] + (1 - alpha)  * gradient[i - 1]
    scale = np.std(x) + 1e-7
    grad_alpha = np.mean((smoothed[:-1] - x[1:]) * gradient[:-1]).astype(float)
    grad_alpha /= scale
    print("grad_alpha", grad_alpha)
    return grad_alpha

def calc_alpha_loss_gradient2(alpha: float, x:np.ndarray, smoothed: np.ndarray,
                             y_pred: np.ndarray) -> float:
    N = len(smoothed)
    gradient = np.zeros_like(smoothed)
    gradient[0] = 0
    for i in range(1, N):
        gradient[i] = x[i] - smoothed[i - 1] + (1 - alpha)  * gradient[i - 1]
    scale = np.std(x) + 1e-7
    grad_alpha = np.mean((smoothed - y_pred) * gradient).astype(float)
    grad_alpha /= scale
    print("grad_alpha", grad_alpha)
    return grad_alpha

def get_smoothed(x: np.ndarray, alpha: float) -> np.ndarray:
    N = len(x)
    smoothed = np.zeros_like(x)
    smoothed[0] = x[0]
    for i in range(1, N):
        smoothed[i] = alpha * x[i] + (1 - alpha) * smoothed[i - 1]
    return smoothed

def search_alpha(x: np.ndarray, n_steps: int = 10) -> float:
    theta = 0
    alpha = alpha_from_theta(theta)
    prev_grad_alpha = 0
    N = len(x)
    lr = 0.01
    for step in range(n_steps):
        print(f'Step {step}: alpha = {alpha}')
        smoothed = get_smoothed(x, alpha)
        train_series, test_series = train_test_split(x)
        train_len, test_len = len(train_series), len(test_series)
        model_fit = fit(train_series, p=5, i=2, q=5)
        train_pred = model_fit.predict(start=0)
        grad_alpha = calc_alpha_loss_gradient2(alpha, x=x[:train_len],
                                              smoothed=smoothed[:train_len], y_pred=train_pred)
        # grad_alpha = calc_alpha_loss_gradient(alpha, x=x[:train_len],
        #                                       smoothed=smoothed[:train_len])
        # theta = theta_from_alpha(alpha)
        grad_theta = grad_alpha * dalpha_dtheta(theta)
        theta -= lr * grad_theta
        alpha = alpha_from_theta(theta)
        # if step > 0:
        #     grad_pct_change = np.abs(grad_alpha - prev_grad_alpha) / prev_grad_alpha
        #     if grad_pct_change < 0.01:
        #         lr *= 3
        #     elif grad_pct_change > 0.1:
        #         lr /= 3
        # prev_grad_alpha = grad_alpha
    print(alpha)
    return alpha

if __name__ == '__main__':
    x = generate_time_series(1000)
    alpha = search_alpha(x, n_steps=30)
    #alpha = 0.615
    # alpha = 0.733
    smoothed = get_smoothed(x, alpha)
    train_series, test_series = train_test_split(x)
    train_len, test_len = len(train_series), len(test_series)
    model_fit = fit(train_series, p=5, i=2, q=5)
    train_pred = model_fit.predict(start=0)
    train_rmse = np.sqrt(np.sum((smoothed[:train_len] - train_pred) ** 2)/train_len)
    print(f"Train RMSE: {train_rmse}")

    test_pred = model_fit.forecast(steps=len(test_series))
    test_rmse = np.sqrt(np.sum((smoothed[train_len:] - test_pred) ** 2) / test_len)
    print(f"Test RMSE: {test_rmse}")
    plt.plot(train_pred, label='train_pred')
    plt.plot(smoothed, label='smoothed')
    plt.plot(np.arange(train_len, train_len+test_len), test_pred, label='test_pred')
    plt.legend()
    #plt.savefig('Smoothed_Tuning.pdf')
    plt.show()