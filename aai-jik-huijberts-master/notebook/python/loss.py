import numpy as np


def mse(pred, targets):
    return np.square(np.subtract(targets, pred)).mean()


def crossentropy(pred, targets, return_mean=True):
    samples = len(targets)
    correct_confidences = None
    # Clip data to prevent division by 0
    y_pred_c = np.clip(targets, 1e-7, 1 - 1e-7)
    if len(pred.shape) == 1:
        correct_confidences = y_pred_c[range(samples), pred]
    elif len(pred.shape) == 2:
        correct_confidences = np.sum(y_pred_c * pred, axis=1)
    loss = -np.log(correct_confidences)
    m_loss = np.mean(loss)
    return m_loss if return_mean else loss

