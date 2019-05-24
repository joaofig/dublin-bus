import numpy as np


def calculate_q(sigma_speed):
    q = np.zeros((4, 4), dtype=np.float)
    q[2, 2] = sigma_speed[0] ** 2
    q[3, 3] = sigma_speed[1] ** 2
    return q


def calculate_r(sigma: np.ndarray) -> np.ndarray:
    r = np.zeros((2, 2), dtype=np.float)
    r[0, 0] = sigma[0] ** 2
    r[1, 1] = sigma[1] ** 2
    return r


def calculate_p(sigma, sigma_speed):
    p = np.zeros((4, 4), dtype=np.float)
    p[0, 0] = sigma[0] ** 2
    p[1, 1] = sigma[1] ** 2
    p[2, 2] = sigma_speed[0] ** 2
    p[3, 3] = sigma_speed[1] ** 2
    return p


def calculate_phi(dt):
    """
    Calculates the Φ matrix
    :param dt: Δtᵢ
    :return: The Φ matrix
    """
    phi = np.eye(4)
    phi[0, 2] = dt
    phi[1, 3] = dt
    return phi


def calculate_kalman_gain(p, c, r):
    num = np.matmul(p, np.transpose(c))
    den = np.matmul(c, num) + r
    return np.matmul(num, np.linalg.pinv(den))


def predict_step(prev_x, prev_p, phi, sigma_speed):
    next_x = np.matmul(phi, prev_x)
    next_p = np.matmul(np.matmul(phi, prev_p), np.transpose(phi)) + calculate_q(sigma_speed)
    return next_x, next_p


def update_step(predicted_x, predicted_p, c, y, sigma):
    r = calculate_r(sigma)
    k = calculate_kalman_gain(predicted_p, c, r)
    updated_x = predicted_x + np.matmul(k, y - np.matmul(c, predicted_x))
    identity = np.eye(4)
    updated_p = np.matmul(identity - np.matmul(k, c), predicted_p)
    return updated_x, updated_p
