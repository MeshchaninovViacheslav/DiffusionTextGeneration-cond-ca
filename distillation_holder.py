import torch
import random

from diffusion_holder import DiffusionRunner

from typing import Optional, Union, Dict, Tuple
from functools import partial

from diffusion_utils.dynamic import DynamicSDE
from diffusion_utils.solvers import create_solver

from utils.util import mse_loss, get_stat, recon_loss, bert_acc, dict_to_cuda, reduce_tensor, set_seed, l1_loss, smooth_l1_loss


class DistillationRunner(DiffusionRunner):
    def calc_loss(
                self,
                clean_x,
                cond_x,
                trg: Optional[Dict] = None,
                cond: Optional[Dict] = None,
                eps: float = 1e-5,
        ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
            mask = None
            if cond is None:
                cond_mask = None
            else:
                cond_mask = cond.get("attention_mask", None)

            # Noizing
            batch_size = clean_x.size(0)

            t = self.sample_time(batch_size, eps=eps)
            # Timestep for boundary condition loss
            t_min = torch.ones_like(t, device=self.device) * self.config.generation.t_min
            x_t_min = self.dynamic.marginal(clean_x, t_min)['x_t']
        
            marg_forward = self.dynamic.marginal(clean_x, t)
            x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']
            x_0_self_cond = torch.zeros_like(clean_x, dtype=clean_x.dtype)


            # Model prediction
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                    scores = self.calc_score(
                        model=self.ddp_score_estimator,
                        x_t=x_t,
                        t=t,
                        cond=cond_x,
                        cond_mask=cond_mask,
                        attention_mask=mask,
                        x_0_self_cond=x_0_self_cond,
                    )
            
            # Model prediction for boundary condition
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                    x_bound = self.calc_score(
                        model=self.ddp_score_estimator,
                        x_t=x_t_min,
                        t=t_min,
                        cond=cond_x,
                        cond_mask=cond_mask,
                        attention_mask=mask,
                        x_0_self_cond=x_0_self_cond,
                    )['x_0']                    
            
            # Consistency target prediction
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16), torch.no_grad():
                next_t = t - (self.dynamic.T - self.config.generation.t_min) / (self.dynamic.N + 1)
                # Making sure t > 0
                next_t[next_t < 0] = t[next_t < 0]

                x_t_1 = self.teacher_solver.step(
                    x_t=x_t, 
                    t=t, 
                    next_t=next_t,
                    cond=cond_x,
                    cond_mask=cond_mask,
                    attention_mask=mask,
                    x_0_self_cond=x_0_self_cond,
                )["x_mean"].detach()

                x_trg = self.calc_score(
                    model=self.ddp_score_estimator,
                    x_t=x_t_1,
                    t=next_t,
                    cond=cond_x,
                    cond_mask=cond_mask,
                    attention_mask=mask,
                    x_0_self_cond=x_0_self_cond
                )['x_0'].detach()
        
            # MSE losses
            x_0 = scores["x_0"]
            loss_x_0 = mse_loss(x_trg, x_0, mask)

            # Boundary condition loss
            boundary_loss = mse_loss(x_bound, x_t_min, mask)

            # Decoder reconstruction
            logits = self.pred_logits(pred_embeddings=x_0)
            ce_loss = recon_loss(logits, trg["input_ids"], mask)


            loss = loss_x_0 #+ 0.25 * boundary_loss

            loss_dict = {
                'loss': loss,
                'loss_x_0': loss_x_0,
                'loss_ce': ce_loss,
                'accuracy': bert_acc(targets=trg["input_ids"], outputs=logits, mask=mask),
                'boundary': boundary_loss
            }

            stat_dict = {}
            clean_x_dict = get_stat(clean_x, mask)
            for key in clean_x_dict:
                stat_dict[f"clean_x_{key}"] = clean_x_dict[key]

            x_0_dict = get_stat(x_0, mask)
            for key in x_0_dict:
                stat_dict[f"x_0_{key}"] = x_0_dict[key]

            mask = trg["attention_mask"]
            clean_x_dict_SPT = get_stat(clean_x, mask)
            for key in clean_x_dict_SPT:
                stat_dict[f"clean_x_woSPT_{key}"] = clean_x_dict_SPT[key]

            x_0_dict_SPT = get_stat(x_0, mask)
            for key in x_0_dict_SPT:
                stat_dict[f"x_0_woSPT_{key}"] = x_0_dict_SPT[key]

            return loss_dict, stat_dict


    @torch.no_grad()
    def pred_embeddings(
            self, batch_size: int,
            cond_x=None,
            cond_mask=None,
            attention_mask=None,
    ) -> torch.Tensor:
        self.score_estimator.eval()
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            self.encoder_gen.encoder.config.hidden_size
        )

        with torch.no_grad():
            x_0_self_cond = torch.zeros(shape, device=self.device)
            x = self.dynamic.prior_sampling(shape).to(self.device)
            pred_embeddings = self.calc_score(model=self.ddp_score_estimator,
                                              x_t=x,
                                              t=torch.ones((1,), device=self.device) * self.dynamic.T,
                                              cond=cond_x,
                                              cond_mask=cond_mask,
                                              attention_mask=attention_mask,
                                              x_0_self_cond=x_0_self_cond)['x_0']

        return pred_embeddings