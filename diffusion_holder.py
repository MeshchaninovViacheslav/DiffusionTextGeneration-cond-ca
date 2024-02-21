import os
import torch
import wandb
import numpy as np
import random
import torch.distributed as dist
from ml_collections import ConfigDict
from typing import Optional, Union, Dict, Tuple
from tqdm.auto import trange
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import defaultdict
from torch.cuda.amp import GradScaler
from torch import Tensor
import json

from diffusion_utils.dynamic import DynamicSDE
from diffusion_utils.solvers import create_solver

from utils.ema_model import ExponentialMovingAverage
from data import create_dataset

from model.score_estimator_cond import ScoreEstimatorEMB
from model import Encoder
from model.enc_normalizer import EncNormalizer
from model.decoder import BertDecoder

from utils import mse_loss, get_stat, recon_loss, bert_acc, dict_to_cuda, reduce_tensor, set_seed, l1_loss, smooth_l1_loss

from estimation_utils.estimate import estimate


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False
    ):
        self.config = config

        # Condition Encoder
        self.tokenizer_cond = AutoTokenizer.from_pretrained(config.model.conditional_encoder_name)
        self.encoder_cond = Encoder(
            config.model.conditional_encoder_name,
            enc_normalizer=None,
            is_change_sp_tokens=False,
        ).cuda()
        if not config.model.conditional_encoder_train:
            self.encoder_cond.eval()

        # Diffusion Encoder
        self.tokenizer_gen = AutoTokenizer.from_pretrained(config.model.encoder_name)
        self.gen_enc_normalizer = EncNormalizer(
            enc_mean_path=self.config.data.enc_gen_mean,
            enc_std_path=self.config.data.enc_gen_std,
        )
        self.encoder_gen = Encoder(
            config.model.encoder_name,
            enc_normalizer=self.gen_enc_normalizer,
            is_change_sp_tokens=True,
        ).eval().cuda()

        # Decoder
        self.decoder = BertDecoder(model_name=config.model.encoder_name, mode=config.model.decoder_mode)
        self.restore_decoder()
        self.decoder = self.decoder.cuda().eval()

        # Score estimator
        self.bert_config = config.bert_config
        self.bert_config.use_self_cond = config.use_self_cond
        self.score_estimator = ScoreEstimatorEMB(
            input_size=self.encoder_gen.encoder.config.hidden_size,
            config=self.config.bert_config
        ).cuda()

        self.ddp_score_estimator = self.score_estimator
        if self.config.ddp:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[dist.get_rank()],
                broadcast_buffers=False,
            )

        # Number of parameters
        self.config.params_number = ConfigDict()
        self.config.params_number.score_estimator = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)
        self.config.params_number.decoder = sum(p.numel() for p in self.decoder.parameters())
        self.config.params_number.conditional_encoder = sum(p.numel() for p in self.encoder_cond.parameters())
        self.config.params_number.generative_encoder = sum(p.numel() for p in self.encoder_gen.parameters())

        self.device = next(self.score_estimator.parameters()).device

        # Dynamic
        self.dynamic = DynamicSDE(config=config)
        self.diff_eq_solver = create_solver(config)(
            dynamic=self.dynamic,
            score_fn=partial(self.calc_score, model=self.ddp_score_estimator),
            ode_sampling=config.training.ode_sampling
        )

        # Datasets
        self.train_datasets_iter = create_dataset(
            dataset_name=config.data.dataset_name,
        )(
            split="train",
            tokenizer_cond=self.tokenizer_cond,
            tokenizer_gen=self.tokenizer_gen,
            max_sequence_len=self.config.data.max_sequence_len,
            max_context_len=self.config.data.max_context_len,
            base_path=config.data.dataset_path,
        ).get_data()
        self.train_dataset = None

        self.valid_datasets_iter = create_dataset(
            dataset_name=config.data.dataset_name,
        )(
            split="valid",
            tokenizer_cond=self.tokenizer_cond,
            tokenizer_gen=self.tokenizer_gen,
            max_sequence_len=self.config.data.max_sequence_len,
            max_context_len=self.config.data.max_context_len,
            base_path=config.data.dataset_path,
        ).get_data()
        self.valid_dataset = next(self.valid_datasets_iter)

        if self.config.ddp and dist.get_rank() == 0:
            wandb.init(
                project=self.config.project_name,
                name=self.config.training.checkpoints_prefix,
                config=dict(self.config),
                mode="online"
            )

        # Checkpoint loading
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), config.model.ema_rate)

        if eval:
            self.restore_parameters(self.device)
            self.switch_to_ema()
            self.score_estimator.eval()
        else:
            self.set_optimizer()
            self.set_scheduler()
            self.set_grad_scaler()
            self.step = 0
            
            if self.load_checkpoint():
                self.estimate()
                self.validate()

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.config.training.checkpoints_folder
        if device is None:
            device = torch.device('cpu')
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix

        ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '.pth')["ema"]
        self.ema.load_state_dict(ema_ckpt)

    def restore_decoder(self):
        self.decoder.load_state_dict(torch.load(self.config.model.decoder_path)["decoder"])

    def save_checkpoint(self, last: bool = False) -> None:
        if not dist.get_rank() == 0:
            return

        if not os.path.exists(self.config.training.checkpoints_folder):
            os.makedirs(self.config.training.checkpoints_folder)
            
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)
        if not os.path.exists(prefix_folder):
            os.makedirs(prefix_folder)

        if last:
            prefix = 'last'
        else:
            prefix = str(self.step)

        save_path = os.path.join(prefix_folder, prefix + ".pth")
        state_dict = {
            "model": self.score_estimator.state_dict(),
            "ema": self.ema.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "scheduler": self.scheduler.state_dict(),
            "scaler": self.grad_scaler.state_dict(),
            "step": self.step,
            "decoder": self.decoder.state_dict(),
        }
        if self.config.is_conditional:
            state_dict["conditional_encoder"] = self.encoder_cond.state_dict()

        torch.save(state_dict, save_path)
        print(f"Save model to: {save_path}")

    def load_checkpoint(self) -> int:
        prefix_folder = os.path.join(self.config.training.checkpoints_folder, self.config.training.checkpoints_prefix)

        if not os.path.exists(prefix_folder):
            return False

        checkpoint_names = list(os.listdir(prefix_folder))
        checkpoint_names = [str(t).replace(".pth", "") for t in checkpoint_names]
        checkpoint_names = [int(t) for t in checkpoint_names if t.isdigit()]

        if not checkpoint_names:
            return False

        name = self.config.training.checkpoint_name
        if not name:
            name = max(checkpoint_names)
        checkpoint_name = f"{prefix_folder}/{name}.pth"

        load = torch.load(checkpoint_name, map_location="cpu")

        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.switch_to_ema()
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.grad_scaler.load_state_dict(load["scaler"])
        self.encoder_cond.load_state_dict(load["conditional_encoder"])
        
        self.step = load["step"]
        if dist.get_rank() == 0:
            print(f"Checkpoint is loaded {checkpoint_name}")
        return True

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.score_estimator
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        parameters = list(self.score_estimator.parameters())
        if self.config.model.conditional_encoder_train:
            parameters += list(self.encoder_cond.parameters())
        optimizer = torch.optim.AdamW(
            parameters,
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay,
            betas=(self.config.optim.beta_1, self.config.optim.beta_2),
            eps=self.config.optim.eps,
        )
        self.warmup = self.config.optim.linear_warmup
        self.grad_clip_norm = self.config.optim.grad_clip_norm
        self.optimizer = optimizer

    def set_scheduler(self) -> None:
        self.scheduler = CosineLRScheduler(
            self.optimizer,
            t_initial=self.config.training.training_iters,
            lr_min=self.config.optim.min_lr,
            warmup_lr_init=self.config.optim.warmup_lr,
            warmup_t=self.config.optim.linear_warmup,
            cycle_limit=1,
            t_in_epochs=False,
        )

    def set_grad_scaler(self) -> None:
        self.grad_scaler = GradScaler()

    def set_train_data_generator(self) -> None:
        del self.train_dataset
        self.train_dataset = next(self.train_datasets_iter)

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size_per_gpu,
            num_workers=30,
            pin_memory=False,
            shuffle=True,
        )

    def set_valid_data_generator(self) -> None:
        self.valid_loader = DataLoader(
            self.valid_dataset,
            batch_size=self.config.validation.batch_size,
            num_workers=10,
            pin_memory=False,
        )
        

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if dist.get_rank() == 0:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor):
        self.optimizer.zero_grad()
        self.grad_scaler.scale(loss).backward()
        self.grad_scaler.unscale_(self.optimizer)

        grad_norm = torch.sqrt(
            sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters() if t.requires_grad]))

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.score_estimator.parameters(),
                max_norm=self.grad_clip_norm
            )

        clipped_grad_norm = torch.sqrt(
            sum([torch.sum(t.grad ** 2) for t in self.score_estimator.parameters() if t.requires_grad]))

        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.grad_scaler.step(self.optimizer)
        self.grad_scaler.update()

        # My custom strategy
        scale = self.grad_scaler._scale.item()
        max_scale = 2 ** 30
        min_scale = 1
        scale = np.clip(scale, min_scale, max_scale)
        self.grad_scaler.update(new_scale=scale)

        self.ema.update(self.score_estimator.parameters())
        self.scheduler.step_update(self.step)
        return grad_norm, clipped_grad_norm

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.cuda.FloatTensor(batch_size).uniform_() * (self.dynamic.T - eps) + eps
    
    def train(self) -> None:
        self.set_valid_data_generator()
        self.train_range = trange(self.step, self.config.training.training_iters, total=self.config.training.training_iters)
        self.train_range.update(self.step)
        self.train_range_iter = iter(self.train_range)

        while True:
            self.set_train_data_generator()
            self.ddp_score_estimator.train()
            self.train_epoch()

            if self.step >= self.config.training.training_iters:
                break

        self.score_estimator.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()

    def train_epoch(self):
        for _, batch in enumerate(self.train_loader):
            if self.step >= self.config.training.training_iters:
                return
            _ = next(self.train_range_iter)

            loss_dict, stat_dict = self.train_step(batch)

            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

            if self.step % self.config.training.eval_freq == 0:
                self.score_estimator.eval()
                self.switch_to_ema()

                self.validate()
                estimate(self)

                self.switch_back_from_ema()
                self.score_estimator.train()
                self.compute_restoration_loss(suffix="valid")
                # self.compute_restoration_loss(suffix="train")

            self.train_range.set_description(
                f"loss_x_0: {loss_dict['loss_x_0'].item():0.4f}, "
                f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
                f"accuracy: {loss_dict['accuracy'].item():0.4f}"
            )

    def train_step(self, batch):
        self.step += 1

        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                if self.config.is_conditional:
                    cond = self.tokenizer_cond(
                        batch["text_src"],
                        add_special_tokens=True,
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                        return_attention_mask=True,
                        return_token_type_ids=False,
                    )
                    cond = dict_to_cuda(cond)
                    cond_x = self.encoder_cond(**{
                        "input_ids": cond["input_ids"],
                        "attention_mask": cond["attention_mask"]
                    })
                else:
                    cond, cond_x = None, None

                trg = self.tokenizer_gen(
                    batch["text_trg"],
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=self.config.data.max_sequence_len,
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )
                trg = dict_to_cuda(trg)
                clean_x = self.encoder_gen(**{
                    "input_ids": trg["input_ids"], 
                    "attention_mask": trg["attention_mask"]
                })

        loss_dict, stat_dict = self.calc_loss(clean_x=clean_x, cond_x=cond_x, trg=trg, cond=cond)

        stat_dict["grad_norm"], stat_dict["clipped_grad_norm"] = self.optimizer_step(loss_dict['total_loss'])
        stat_dict["scale_factor"] = torch.Tensor([self.grad_scaler._scale])

        if self.step % 10 == 0:
            stat_dict["weight_norm"] = torch.sqrt(
                sum([torch.sum(t.data ** 2) for t in self.score_estimator.parameters()]))

            for k, v in loss_dict.items():
                self.log_metric(k, 'train', v.item())

            for k, v in stat_dict.items():
                self.log_metric('statistics', k, v.item())

        return loss_dict, stat_dict

    def validate(self) -> None:
        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        with torch.no_grad():
            for batch in self.valid_loader:
                if self.config.is_conditional:
                    cond = self.tokenizer_cond(
                        batch["text_src"],
                        add_special_tokens=True,
                        padding=True,
                        truncation=False,
                        return_tensors="pt",
                        return_attention_mask=True,
                    )
                    cond = dict_to_cuda(cond)
                    cond_x = self.encoder_cond(**{
                        "input_ids": cond["input_ids"],
                        "attention_mask": cond["attention_mask"]
                    })
                else:
                    cond, cond_x = None, None

                trg = self.tokenizer_gen(
                    batch["text_trg"],
                    add_special_tokens=True,
                    padding="max_length",
                    max_length=self.config.data.max_sequence_len,
                    truncation=True,
                    return_tensors="pt",
                    return_attention_mask=True,
                )
                trg = dict_to_cuda(trg)
                clean_x = self.encoder_gen(**{
                    "input_ids": trg["input_ids"], 
                    "attention_mask": trg["attention_mask"]
                }) 

                loss_dict, _ = self.calc_loss(clean_x=clean_x, cond_x=cond_x, trg=trg, cond=cond)
                for k, v in loss_dict.items():
                    if k in valid_loss:
                        valid_loss[k] += v.item() * clean_x.size(0)
                    else:
                        valid_loss[k] = torch.Tensor([v.item() * clean_x.size(0)])
                valid_count += clean_x.size(0)

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        for k, v in valid_loss.items():
            self.log_metric(k, 'valid_loader', v)

    def calc_score(
            self,
            model,
            x_t, t,
            cond=None,
            attention_mask=None,
            cond_mask=None,
            x_0_self_cond=None
    ) -> Dict[str, torch.Tensor]:
        """
        x_0 - prediction x_0(x_t, t)
        eps = (x_t - sqrt(alpha_t) * x_0) / std
        score = (-x_t + sqrt(alpha_t) * x_0) / std**2
        """
        params = self.dynamic.marginal_params(t)
        #t = torch.clip(t + self.delta, max=self.dynamic.T)
        x_0 = model(
            x_t=x_t, time_t=t, cond=cond,
            attention_mask=attention_mask, cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond
        )

        eps_theta = (x_t - params["mu"] * x_0) / params["std"]
        score = -eps_theta / params["std"]
        return {
            "score": score,
            "x_0": x_0,
            "eps_theta": eps_theta
        }

    def calc_loss(
            self,
            clean_x: Tensor,
            cond_x: Tensor,
            trg: Optional[Dict] = None,
            cond: Optional[Dict] = None,
            eps: float = 1e-5,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mask = None  # X["input_mask"]
        if cond is None:
            cond_mask = None
        else:
            cond_mask = cond.get("attention_mask", None)

        # Noizing
        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size, eps=eps)
        marg_forward = self.dynamic.marginal(clean_x, t)
        x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(clean_x, dtype=clean_x.dtype)
        if self.config.use_self_cond and random.random() > 0.5:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                t_next = t
                params_next = self.dynamic.marginal_params(t_next)
                x_t_next = params_next["mu"] * clean_x + params_next["std"] * noise

                with torch.no_grad():
                    x_0_self_cond = self.ddp_score_estimator(
                        x_t=x_t_next, time_t=t_next, cond=cond_x,
                        attention_mask=mask, cond_mask=cond_mask,
                        x_0_self_cond=x_0_self_cond
                    ).detach()

        # model prediction
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

        # MSE losses
        x_0 = scores["x_0"]
        loss_x_0 = mse_loss(clean_x, x_0, mask)

        # Decoder reconstruction
        logits = self.pred_logits(pred_embeddings=x_0)
        ce_loss = recon_loss(logits, trg["input_ids"], mask)

        loss = loss_x_0
        loss_dict = {
            'loss': loss,
            'loss_x_0': loss_x_0,
            'loss_ce': ce_loss,
            'accuracy': bert_acc(targets=trg["input_ids"], outputs=logits, mask=mask)
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
    def generate_text(self, num_texts):
        self.set_valid_data_generator()

        result_dict = {
            "GEN": [],
            "GT": []
        }
        if self.config.is_conditional:
            result_dict["COND"] = []

        for batch in self.valid_loader:
            tmp_batch_size = int(min(len(batch["text_src"]), num_texts - len(result_dict["GEN"])))
            
            for key in batch:
                batch[key] = batch[key][:tmp_batch_size]

            if cond is None:
                cond = None
            else:
                cond = self.tokenizer_cond(
                    batch["text_src"],
                    add_special_tokens=True,
                    padding=True,
                    truncation=False,
                    return_tensors="pt",
                    return_attention_mask=True,
                    return_token_type_ids=False,
                )
                cond = dict_to_cuda(cond)
           
            gen_text = self.generate_text_batch(
                batch_size=tmp_batch_size,
                cond=cond,
                attention_mask=None,
            )[0]
    
            result_dict["GEN"] += gen_text
            result_dict["GT"] += batch["text_trg"]
            if self.config.is_conditional:
                result_dict["COND"] += batch["text_src"]

            if len(result_dict["GEN"]) >= num_texts:
                break
        return result_dict

    @torch.no_grad()
    def generate_text_batch(self, batch_size, cond=None, attention_mask=None):
        if attention_mask is not None:
            attention_mask = attention_mask.cuda()

        if cond is None:
            cond_x = None
            cond_mask = None
        else:
            cond_x = self.encoder_cond(input_ids=cond["input_ids"], attention_mask=cond["attention_mask"])
            cond_mask = cond["attention_mask"]

        pred_embeddings = self.pred_embeddings(
            batch_size,
            attention_mask=attention_mask,
            cond_x=cond_x,
            cond_mask=cond_mask
        )

        output = self.pred_logits(pred_embeddings)
        tokens = output.argmax(dim=-1)

        eos_id = self.tokenizer_gen.vocab[self.tokenizer_gen.sep_token]

        tokens = tokens.detach().cpu().tolist()
        tokens_list = []
        for seq in tokens:
            id = 0
            while id < len(seq) and seq[id] != eos_id:
                id += 1
            tokens_list.append(seq[0: id])

        text = self.tokenizer_gen.batch_decode(tokens_list, skip_special_tokens=True)
        return text, pred_embeddings

    @torch.no_grad()
    def pred_logits(self, pred_embeddings):
        pred_embeddings = self.gen_enc_normalizer.denormalize(pred_embeddings)
        output = self.decoder(pred_embeddings)
        return output

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
            x = self.dynamic.prior_sampling(shape).to(self.device)
            x_0_self_cond = torch.zeros_like(x, dtype=x.dtype)
            eps_t = self.config.generation.t_min

            timesteps = torch.linspace(self.dynamic.T, eps_t, self.dynamic.N + 1, device=self.device)

            for idx in tqdm(range(self.dynamic.N)):
                t = timesteps[idx]
                next_t = timesteps[idx + 1]

                input_t = t * torch.ones(shape[0], device=self.device)
                next_input_t = next_t * torch.ones(shape[0], device=self.device)

                output = self.diff_eq_solver.step(
                    x_t=x, t=input_t, next_t=next_input_t,
                    cond=cond_x,
                    cond_mask=cond_mask,
                    attention_mask=attention_mask,
                    x_0_self_cond=x_0_self_cond,
                )

                x, x_mean = output["x"], output["x_mean"]
                x_0_self_cond = output["x_0"]

            pred_embeddings = x_mean

        return pred_embeddings
