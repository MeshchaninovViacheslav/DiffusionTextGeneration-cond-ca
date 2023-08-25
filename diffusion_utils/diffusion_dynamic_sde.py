import torch
import numpy as np
from typing import Dict, Union, List, Optional

from diffusion_utils.schedulers import cosine, linear, quadratic


class DDPM_SDE:
    def __init__(self, config):
        """Construct a Variance Preserving SDE.
        Args:
          beta_min: value of beta(0)
          beta_max: value of beta(1)
          N: number of discretization steps
        """
        self.N = config.sde.N
        self.beta_0 = config.sde.beta_min
        self.beta_1 = config.sde.beta_max
        self.prediction = config.model.prediction
        self.scheduler = config.sde.scheduler

    @property
    def T(self):
        return 1

    def sde(self, x, t) -> Dict[str, torch.Tensor]:
        """
        sde: dx = drift * dt + diffusion * dw
        drift = -1/2 * beta * x_t
        diffusion = sqrt(beta)
        """
        beta_t = self.scheduler.beta_t(t)
        drift = -0.5 * beta_t[:, None, None].to(x.device) * x
        diffusion = torch.sqrt(beta_t).to(x.device)
        return {
            "drift": drift,
            "diffusion": diffusion
        }

    def marginal_params_tensor(self, x, t) -> Dict[str, torch.Tensor]:
        """
        x_t = x_0 * alpha + eps * std
        beta(s) = (beta_max - beta_min) * s + beta_min
        alpha_real = exp(-integrate(beta(s) ds)) = exp(-1/2 * (beta_max - beta_min) * t**2 - beta_min * t)
        here alpha = sqrt(alpha_real) in order to multiply x without sqrt
        std = sqrt(1 - alpha_real)
        """
        alpha, std = self.scheduler.alpha_std(t)
        return {
            "alpha": alpha.to(x.device),
            "std": std.to(x.device)
        }

    def marginal_prob(self, x, t) -> Dict[str, torch.Tensor]:
        params = self.marginal_params_tensor(x, t)
        alpha, std = params['alpha'], params['std']
        mean = alpha * x
        return {
            "mean": mean,
            "std": std
        }

    def prob_flow(self, model, x_t, t):
        sde_params = self.sde(x_t, t)
        drift, diffusion = sde_params['drift'], sde_params['diffusion']
        scores = self.calc_score(model, x_t, t)
        score = scores['score']
        drift = drift - diffusion[:, None, None] ** 2 * score * 0.5
        return drift

    def marginal_forward(self, x, t) -> Dict[str, torch.Tensor]:
        params = self.marginal_params_tensor(x, t)
        alpha, std = params['alpha'], params['std']
        mean = alpha * x
        noise = torch.randn_like(mean)
        return {
            "mean": mean,
            "x_t": mean + noise * std,
            "noise": noise,
            "noise_t": noise * std,
            "score": -noise / std,
        }

    # def __marginal_params_scalar(self, t: float) -> Dict[str, torch.Tensor]:
    #     """
    #     x_t = x_0 * alpha + eps * std
    #     beta(s) = (beta_max - beta_min) * s + beta_min
    #     alpha_real = exp(-integrate(beta(s) ds)) = exp(-1/2 * (beta_max - beta_min) * t**2 - beta_min * t)
    #     here alpha = sqrt(alpha_real) in order to multiply x without sqrt
    #     std = sqrt(1 - alpha_real)
    #     """
    #     log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
    #     log_gamma_coeff = log_mean_coeff * 2
    #     alpha = torch.exp(log_mean_coeff)
    #     std = torch.sqrt(1. - torch.exp(log_gamma_coeff))
    #     return {
    #         "alpha": alpha,
    #         "std": std
    #     }
    #
    # def _-marginal_std(self, t: torch.Tensor) -> torch.Tensor:
    #     log_gamma_coeff = -0.5 * t ** 2 * (self.beta_1 - self.beta_0) - t * self.beta_0
    #     std = torch.sqrt(1. - torch.exp(log_gamma_coeff)).to(t.device)[:, None, None]
    #     return std

    def prior_sampling(self, shape) -> torch.Tensor:
        return torch.randn(*shape)

    def reverse(self, ode_sampling=False) -> "RSDE Class":
        """Create the reverse-time SDE/ODE.
        Args:
          score_fn: A time-dependent score-based model that takes x and t and returns the score.
          ode_sampling: If `True`, create the reverse-time ODE used for probability flow sampling.
        """
        N = self.N
        T = self.T
        sde_cls = self

        # Build the class for reverse-time SDE.
        class RSDE:
            def __init__(self):
                self.N = N
                self.ode_sampling = ode_sampling

            @property
            def T(self):
                return T

            def sde(self, model, x_t, t, cond=None, attention_mask=None, cond_mask=None) -> Dict[str, torch.Tensor]:
                """Create the drift and diffusion functions for the reverse SDE/ODE.
                SDE:
                    dx = (-1/2 * beta * x_t - beta * score) * dt + sqrt(beta) * dw
                ODE:
                    dx = (-1/2 * beta * x_t - 1/2 * beta * score) * dt
                """
                sde_params = sde_cls.sde(x_t, t)
                drift_par, diffusion = sde_params['drift'], sde_params['diffusion']  # -1/2 * beta * x_t, sqrt(beta)

                scores = sde_cls.calc_score(model, x_t, t, cond=cond, attention_mask=attention_mask,
                                            cond_mask=cond_mask)
                score = scores['score']
                drift = drift_par - diffusion[:, None, None] ** 2 * score * (0.5 if self.ode_sampling else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.ode_sampling else diffusion
                return {
                    "score": score,
                    "drift": drift,
                    "drift_par": drift_par,
                    "diffusion": diffusion,
                    "x_0": scores["x_0"]
                }

        return RSDE()

    def calc_score(self, model, x_t, t, cond=None, attention_mask=None, cond_mask=None) -> Dict[str, torch.Tensor]:
        """
        x_0 - prediction x_0(x_t, t)
        eps = (x_t - sqrt(alpha_t) * x_0) / std
        score = (-x_t + sqrt(alpha_t) * x_0) / std**2
        """
        params = self.marginal_params_tensor(x_t, t)
        x_0 = model(x_t=x_t, time_t=t, cond=cond, attention_mask=attention_mask, cond_mask=cond_mask)
        eps_theta = (x_t - params["alpha"] * x_0) / params["std"]
        score = -eps_theta / params["std"]
        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }


class EulerDiffEqSolver:
    def __init__(self, sde: DDPM_SDE, ode_sampling=False):
        self.sde: DDPM_SDE = sde
        self.ode_sampling: bool = ode_sampling
        self.rsde = sde.reverse(ode_sampling)

    def step(self, model, x_t, t, **kwargs) -> Dict[str, torch.Tensor]:
        dt = -1. / self.rsde.N
        z = torch.randn_like(x_t)
        rsde_params = self.rsde.sde(model, x_t, t, **kwargs)
        drift, diffusion = rsde_params['drift'], rsde_params['diffusion']
        x_mean = x_t + drift * dt
        if not self.ode_sampling:
            x = x_mean + diffusion[:, None, None] * np.sqrt(-dt) * z
        else:
            x = x_mean
        return {
            "x": x,
            "x_mean": x_mean,
            "score": rsde_params['score'],
            "x_0": rsde_params['x_0'],
            "diffusion": diffusion,
            "drift": drift,
            "drift_par": rsde_params["drift_par"]
        }


def create_sde(config):
    possible_sde = {
        "vp-sde": DDPM_SDE,
    }
    return possible_sde[config.sde.typename](config)


def create_solver(config, *solver_args, **solver_kwargs):
    possible_solver = {
        "euler": EulerDiffEqSolver,
    }
    return possible_solver[config.sde.solver](*solver_args, **solver_kwargs)
