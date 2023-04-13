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


class Cosine_lin(Scheduler):
    def __init__(self, beta_0, beta_1):
        self.beta_0 = beta_0
        self.beta_1 = beta_1
        self.eps_t = 1e-3

    def beta_t(self, t):
        return self.beta_0 + (self.beta_1 - self.beta_0) * t

    def alpha_std(self, t):
        t = t[:, None, None]
        alpha_sqrt = 1 - t
        std = torch.sqrt(1 - torch.square(alpha_sqrt))
        return alpha_sqrt, std

    def reverse(self, alpha):
        t = (-1 / 2 * self.beta_0 + np.sqrt(
            (1 / 2 * self.beta_0) ** 2 - (self.beta_1 - self.beta_0) * np.log(alpha))) / (
                    1 / 2 * (self.beta_1 - self.beta_0))
        return t

    def clip(self, t):
        return torch.clip(t, min=self.eps_t, max=1 - self.eps_t)


class Linear(Scheduler):
    def __init__(self, eps_t=1e-4):
        self.eps_t = eps_t

    def beta_t(self, t):
        t = self.clip(t)
        beta_t = 1 / (1 - t)
        beta_t = torch.clip(beta_t, min=0, max=20)
        return beta_t

    def alpha_std(self, t):
        t = self.clip(t)
        t = t[:, None, None]
        alpha_sqrt = 1 - t
        std = torch.sqrt(1 - torch.square(alpha_sqrt))
        return alpha_sqrt, std

    def reverse(self, alpha):
        return 1 - alpha

    def clip(self, t):
        return torch.clip(t, min=self.eps_t, max=1 - self.eps_t)


class Quadratic(Scheduler):
    def __init__(self, a=2, eps_t=1e-4):
        self.eps_t = eps_t
        self.a = a

    def beta_t(self, t):
        t = self.clip(t)
        beta_t = self.a / (1 - t)
        # beta_t = torch.clip(beta_t, min=0, max=20)
        return beta_t

    def alpha_std(self, t):
        t = self.clip(t)
        t = t[:, None, None]
        alpha_sqrt = (1 - t) ** 2
        std = torch.sqrt(1 - torch.square(alpha_sqrt))
        return alpha_sqrt, std

    def reverse(self, alpha):
        return 1 - torch.sqrt(alpha)

    def clip(self, t):
        return torch.clip(t, min=self.eps_t, max=1 - self.eps_t)


class Exponential(Scheduler):
    def __init__(self, a, eps_t=1e-4):
        self.eps_t = eps_t
        self.clip_coef = 20.
        self.a = a

    def beta_t(self, t):
        t = self.clip(t)
        beta_t = self.a / (1 - t)
        return beta_t

    def alpha_std(self, t):
        t = self.clip(t)
        t = t[:, None, None]
        t_thr = 1 - self.a / self.clip_coef
        indicator = (t < t_thr).int()
        alpha_sqrt = (1 - t) ** self.a
        # alpha_sqrt = indicator * ((1 - t) ** self.a) + \
        #              (1 - indicator) * ((1 - t_thr) ** self.a * torch.exp((t_thr - t) * self.clip_coef))
        # alpha_sqrt = torch.clip(alpha_sqrt, 0, 1)
        std = torch.sqrt(1 - torch.square(alpha_sqrt))
        return alpha_sqrt, std

    def reverse(self, alpha):
        pass

    def clip(self, t):
        return torch.clip(t, min=self.eps_t, max=1 - self.eps_t)
