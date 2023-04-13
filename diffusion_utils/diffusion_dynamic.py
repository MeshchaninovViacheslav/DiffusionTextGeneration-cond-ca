import torch
import numpy as np
from typing import Dict, Union, List, Optional


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

    @property
    def T(self):
        return 1

    def sde(self, x, t) -> Dict[str, torch.Tensor]:
        beta_t = self.beta_0 + t * (self.beta_1 - self.beta_0)
        drift = -0.5 * beta_t[:, None, None].to(x.device) * x
        diffusion = torch.sqrt(beta_t).to(x.device)
        return {
            "drift": drift,
            "diffusion": diffusion
        }

    def marginal_params_tensor(self, x, t) -> Dict[str, torch.Tensor]:
        """
        x_t = x_0 * alpha + eps * std
        """
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_gamma_coeff = log_mean_coeff * 2
        alpha = torch.exp(log_mean_coeff[:, None, None]).to(x.device)
        std = torch.sqrt(1. - torch.exp(log_gamma_coeff)).to(x.device)[:, None, None]
        return {
            "alpha": alpha,
            "std": std
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
            "noise_t": noise * std
        }

    def marginal_params_scalar(self, t: float) -> Dict[str, torch.Tensor]:
        log_mean_coeff = -0.25 * t ** 2 * (self.beta_1 - self.beta_0) - 0.5 * t * self.beta_0
        log_gamma_coeff = log_mean_coeff * 2
        alpha = np.exp(log_mean_coeff)
        std = np.sqrt(1. - np.exp(log_gamma_coeff))
        return {
            "alpha": alpha,
            "std": std
        }

    def marginal_std(self, t: torch.Tensor) -> torch.Tensor:
        log_gamma_coeff = -0.5 * t ** 2 * (self.beta_1 - self.beta_0) - t * self.beta_0
        std = torch.sqrt(1. - torch.exp(log_gamma_coeff)).to(t.device)[:, None, None]
        return std

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

            def sde(self, model, x_t, t) -> Dict[str, torch.Tensor]:
                """Create the drift and diffusion functions for the reverse SDE/ODE.
                    dx = (-1/2 * beta * x_t - beta * score) * dt + sqrt(beta) * eps *

                    drift = -1/2 * beta * x_t - beta * score
                    diffusion = sqrt(beta)
                """
                sde_params = sde_cls.sde(x_t, t)
                drift, diffusion = sde_params['drift'], sde_params['diffusion']

                scores = sde_cls.calc_score(model, x_t, t)
                score = scores['score']

                drift = drift - diffusion[:, None, None] ** 2 * score * (0.5 if self.ode_sampling else 1.)
                # Set the diffusion function to zero for ODEs.
                diffusion = 0. if self.ode_sampling else diffusion
                return {
                    "score": score,
                    "drift": drift,
                    "diffusion": diffusion,
                    "x_0": scores["x_0"]
                }

        return RSDE()

    def calc_score(self, model, x_t, t, mask=None) -> Dict[str, torch.Tensor]:
        # eps_theta = model(x_t, t)
        # std = self.marginal_std(t)
        # score = -eps_theta / std
        # return {
        #     "score": score,
        #     "eps_theta": eps_theta
        # }
        x_0 = model(x_t, t, attention_mask=mask)
        params = self.marginal_params_tensor(x_0, t)
        # print(f"alpha: {params['alpha'][0, 0].item()}, "
        #       f"std: {params['std'][0, 0].item()}, "
        #       f"score wo std: {torch.mean(x_t - params['alpha'] * x_0)}")
        score = -(x_t - params["alpha"] * x_0) / (params["std"] ** 2)
        return {
            "score": score,
            "x_0": x_0
        }


class EulerDiffEqSolver:
    def __init__(self, sde: DDPM_SDE, ode_sampling=False):
        self.sde: DDPM_SDE = sde
        self.ode_sampling: bool = ode_sampling
        self.rsde = sde.reverse(ode_sampling)

    def step(self, model, x_t, t) -> Dict[str, torch.Tensor]:
        dt = -1. / self.rsde.N
        z = torch.randn_like(x_t)
        rsde_params = self.rsde.sde(model, x_t, t)
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
            "drift": drift
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
