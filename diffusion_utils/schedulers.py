import torch
import numpy as np
from abc import ABCMeta, abstractmethod


def cosine(t, beta_0, beta_1):
    if t.shape:
        t = t[:, None, None]
    log_mean_coeff = -0.25 * t ** 2 * (beta_1 - beta_0) - 0.5 * t * beta_0
    log_gamma_coeff = log_mean_coeff * 2
    alpha = torch.exp(log_mean_coeff)
    std = torch.sqrt(1. - torch.exp(log_gamma_coeff))
    return alpha, std


def cosine_rev(alpha, beta_0, beta_1):
    t = (-1 / 2 * beta_0 + np.sqrt((1 / 2 * beta_0) ** 2 - (beta_1 - beta_0) * np.log(alpha))) / (
            1 / 2 * (beta_1 - beta_0))
    return t


def linear(t):
    # t = t[:, None, None]
    alpha_sqrt = 1 - t
    std = torch.sqrt(1 - torch.square(alpha_sqrt))
    return alpha_sqrt, std


def linear_rev(alpha):
    return 1 - alpha


def quadratic(t):
    t = t[:, None, None]
    alpha_sqrt = ((1 - t) ** 2)
    std = torch.sqrt(1 - torch.square(alpha_sqrt))
    return alpha_sqrt, std


def quadratic_rev(alpha):
    return 1 - np.sqrt(alpha)


def linear_exp(t, beta_0, beta_1):
    beta = beta_0 ** 2 / (beta_1 - beta_0) + 2 * beta_0 / (beta_1 - beta_0) * np.sqrt(
        beta_0 ** 2 / 4 - (beta_1 - beta_0) * np.log(1 - t))
    if not t.shape:
        t = t[:, None, None]
    alpha_sqrt = (1 - t) * np.exp(-beta)
    std = torch.sqrt(1 - torch.square(alpha_sqrt))
    return alpha_sqrt, std


class Scheduler(metaclass=ABCMeta):
    @abstractmethod
    def beta_t(self, t):
        pass

    @abstractmethod
    def alpha_std(self, t):
        pass

    def reverse(self, alpha):
        pass


class Cosine(Scheduler):
    def __init__(self, beta_0, beta_1):
        self.beta_0 = beta_0
        self.beta_1 = beta_1

    def beta_t(self, t):
        return self.beta_0 + (self.beta_1 - self.beta_0) * t

    def alpha_std(self, t):
        t = t[:, None, None]
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_gamma_coeff = log_mean_coeff * 2
        alpha = torch.exp(log_mean_coeff)
        std = torch.sqrt(1. - torch.exp(log_gamma_coeff))
        return alpha, std

    def reverse(self, alpha):
        t = (-1 / 2 * self.beta_0 + np.sqrt(
            (1 / 2 * self.beta_0) ** 2 - (self.beta_1 - self.beta_0) * np.log(alpha))) / (
                    1 / 2 * (self.beta_1 - self.beta_0))
        return t

class CosineSD(Scheduler):
    def __init__(self, d=1):
        self.d = d
        self.t_thr = 0.95

    def beta_t(self, t):
        t = torch.clip(t, 0, self.t_thr)
        tan = torch.tan(np.pi * t / 2)
        beta_t = np.pi * self.d ** 2 * tan * (1 + tan ** 2) / (1 + self.d ** 2 * tan ** 2)
        return beta_t

    def alpha_std(self, t):
        t = t[:, None, None]
        tan = torch.tan(np.pi * t / 2)
        alpha_t = 1 / torch.sqrt(1 + tan ** 2 * self.d ** 2)
        std_t = torch.sqrt(1 - alpha_t ** 2)
        return torch.clip(alpha_t, 0, 1), torch.clip(std_t, 0, 1)
