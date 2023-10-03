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
from transformers import BertTokenizerFast, T5TokenizerFast
from tqdm import tqdm
from functools import partial
from copy import deepcopy
from timm.scheduler.cosine_lr import CosineLRScheduler
from collections import defaultdict
from torch.cuda.amp import GradScaler
import json

from diffusion_utils.dynamic import DynamicSDE
from diffusion_utils.solvers import create_solver

from utils.ema_model import ExponentialMovingAverage
from data.dataset import create_dataset

from model.score_estimator_cond import ScoreEstimatorEMB
from model.t5_encoder import T5EncoderModel
from model.bert_encoder import BertEncoderModel
from model.enc_normalizer import EncNormalizer
from model.decoder import Decoder

from utils.util import mse_loss, get_stat, recon_loss, bert_acc, dict_to_cuda, reduce_tensor, set_seed

from estimation_utils.util import estimate_model, gather_texts, reduce_metrics, reduce_sum_metrics

from estimation_utils.metrics import BloomMetricConditional, RobertaMetric


class DiffusionRunner:
    def __init__(
            self,
            config: ConfigDict,
            eval: bool = False,
            latent_mode: str = "embeddings"
    ):
        self.config = config
        self.latent_mode = latent_mode
        self.eval = eval
        self.use_self_cond = config.use_self_cond
        self.checkpoints_folder = config.training.checkpoints_folder

        # Encoder for condition

        t5_cfg = "t5-base"
        self.tokenizer_cond = T5TokenizerFast.from_pretrained(t5_cfg)
        self.t5_enc_normalizer = EncNormalizer(
            enc_mean_path=self.config.data.enc_t5_mean,
            enc_std_path=self.config.data.enc_t5_std,
        )
        self.encoder_cond = T5EncoderModel.from_pretrained(
            t5_cfg, enc_normalizer=self.t5_enc_normalizer
        ).eval().cuda()

        bert_cfg = "bert-base-uncased"
        self.tokenizer_gen = BertTokenizerFast.from_pretrained(bert_cfg)
        self.gen_enc_normalizer = EncNormalizer(
            enc_mean_path=self.config.data.enc_bert_mean,
            enc_std_path=self.config.data.enc_bert_std,
        )
        self.encoder_gen = BertEncoderModel.from_pretrained(
            bert_cfg,
            enc_normalizer=self.gen_enc_normalizer
        ).eval().cuda()

        #
        bert_cfg = "bert-base-uncased"
        self.tokenizer_bert = BertTokenizerFast.from_pretrained(bert_cfg)

        # self.decoder = Decoder(
        #     input_size=self.encoder_gen.config.hidden_size,
        #     hidden_size=self.encoder_gen.config.hidden_size,
        #     vocab_size=self.encoder_gen.config.vocab_size
        # )
        self.decoder = self.encoder_gen.cls.cpu()
        self.restore_decoder()
        self.decoder = self.decoder.cuda().eval()

        self.optimizer = None
        self.scheduler = None
        self.step = 0

        # self.load_sde()
        self.bert_config = config.bert_config
        self.bert_config.use_self_cond = config.use_self_cond
        self.score_estimator = ScoreEstimatorEMB(
            input_size=self.encoder_gen.config.hidden_size,
            config=self.bert_config
        ).cuda()

        self.ddp_score_estimator = self.score_estimator
        if self.config.ddp:
            self.ddp_score_estimator = torch.nn.parallel.DistributedDataParallel(
                self.score_estimator,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
            )
        self.total_number_params = sum(p.numel() for p in self.score_estimator.parameters() if p.requires_grad)
        self.config.model.total_number_params = self.total_number_params
        self.device = next(self.score_estimator.parameters()).device

        self.dynamic = DynamicSDE(config=config)
        self.diff_eq_solver = create_solver(config)(
            dynamic=self.dynamic,
            score_fn=partial(self.calc_score, model=self.ddp_score_estimator),
            ode_sampling=config.training.ode_sampling
        )

        if eval:
            self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), config.model.ema_rate)
            self.restore_parameters(self.device)
            self.switch_to_ema()
            self.score_estimator.eval()

        self.grad_expl_dict = defaultdict(list)

        self.train_datasets_iter = create_dataset(
            dataset_name=config.model.dataset,
            downstream_task=config.model.downstream_task
        )(
            split="train",
            tokenizer_bert=self.tokenizer_bert,
            tokenizer_cond=self.tokenizer_cond,
            tokenizer_gen=self.tokenizer_gen,
            max_sequence_len=self.config.data.max_sequence_len,
            pos_begin=self.config.data.pos_begin,
            pos_end=self.config.data.pos_end,
        ).get_data()
        self.train_dataset = None

        self.valid_datasets_iter = create_dataset(
            dataset_name=config.model.dataset,
            downstream_task=config.model.downstream_task
        )(
            split="valid",
            tokenizer_bert=self.tokenizer_bert,
            tokenizer_cond=self.tokenizer_cond,
            tokenizer_gen=self.tokenizer_gen,
            max_sequence_len=self.config.data.max_sequence_len,
            pos_begin=self.config.data.pos_begin,
            pos_end=self.config.data.pos_end,
        ).get_data()
        self.valid_dataset = next(self.valid_datasets_iter)

        if self.config.ddp and dist.get_rank() == 0:
            wandb.init(
                project=self.config.project_name,
                name=self.config.checkpoints_prefix,
                config=dict(self.config),
                mode="online"
            )

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        if device is None:
            device = torch.device('cpu')
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix

        ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '.pth')["ema"]
        self.ema.load_state_dict(ema_ckpt)

    def restore_decoder(self):
        decoder_path = self.config.model.decoder_path
        self.decoder.load_state_dict(torch.load(os.path.join(self.checkpoints_folder, decoder_path))["decoder"])

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
        if self.optimizer is None:
            optimizer = torch.optim.AdamW(
                self.score_estimator.parameters(),
                lr=self.config.optim.lr,
                weight_decay=self.config.optim.weight_decay,
                betas=(self.config.optim.beta_1, self.config.optim.beta_2),
                eps=self.config.optim.eps,
            )
            self.warmup = self.config.optim.linear_warmup
            self.grad_clip_norm = self.config.optim.grad_clip_norm
            self.optimizer = optimizer
        else:
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = self.config.optim.lr
                param_group['weight_decay'] = self.config.optim.weight_decay

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
        print("Dataset length:", len(self.train_dataset))

        self.train_loader = DataLoader(
            self.train_dataset,
            batch_size=self.config.training.batch_size_per_gpu,
            num_workers=30,
            pin_memory=False,
            shuffle=True,
        )

    def set_valid_data_generator(self) -> None:
        if self.config.ddp:
            sampler_valid = torch.utils.data.distributed.DistributedSampler(
                self.valid_dataset,
                shuffle=False
            )
        else:
            sampler_valid = None

        self.valid_loader = DataLoader(
            self.valid_dataset,
            sampler=sampler_valid,
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
            clean_x,
            cond=None,
            X=None,
            eps: float = 1e-5,
    ) -> Tuple[Dict[str, torch.Tensor], Dict[str, torch.Tensor]]:
        mask = None  # X["input_mask"]

        # Noizing
        batch_size = clean_x.size(0)

        t = self.sample_time(batch_size, eps=eps)
        marg_forward = self.dynamic.marginal(clean_x, t)
        x_t, noise, score_clean = marg_forward['x_t'], marg_forward['noise'], marg_forward['score']

        # self-cond estimate
        x_0_self_cond = torch.zeros_like(clean_x, dtype=clean_x.dtype)
        if self.use_self_cond and random.random() > 0.5:
            with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                t_next = torch.clip(t + 1 / self.config.dynamic.N, max=self.dynamic.T)
                params_next = self.marginal_params(t_next)
                x_t_next = params_next["mu"] * clean_x + params_next["std"] * noise

                with torch.no_grad():
                    x_0_self_cond = self.ddp_score_estimator(
                        x_t=x_t_next, time_t=t_next, cond=cond,
                        attention_mask=mask, cond_mask=X["cond_mask"],
                        x_0_self_cond=x_0_self_cond
                    ).detach()

        # model prediction
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            scores = self.calc_score(
                model=self.ddp_score_estimator,
                x_t=x_t,
                t=t,
                cond=cond,
                cond_mask=X["cond_mask"],
                attention_mask=mask,
                x_0_self_cond=x_0_self_cond,
            )

        # MSE losses
        x_0, eps_theta, score = scores["x_0"], scores['eps_theta'], scores["score"]

        loss_x_0 = mse_loss(clean_x, x_0, mask)
        loss_eps = mse_loss(noise, eps_theta, mask)
        loss_score = mse_loss(score_clean, score, mask)

        # Decoder reconstruction
        logits = self.pred_logits(pred_embeddings=x_0, input_ids=X["input_ids"])
        ce_loss = recon_loss(logits, X["input_ids"], mask)

        loss = loss_x_0

        loss = loss + ce_loss * self.config.loss.ce_coef
        loss_dict = {
            'loss': loss,
            'total_loss': loss,
            'loss_eps': loss_eps,
            'loss_x_0': loss_x_0,
            'loss_score': loss_score,
            'loss_ce': ce_loss,
            'accuracy': bert_acc(targets=X["input_ids"], outputs=logits, mask=mask)
        }

        stat_dict = {}
        clean_x_dict = get_stat(clean_x, mask)
        for key in clean_x_dict:
            stat_dict[f"clean_x_{key}"] = clean_x_dict[key]

        x_0_dict = get_stat(x_0, mask)
        for key in x_0_dict:
            stat_dict[f"x_0_{key}"] = x_0_dict[key]

        mask = X["input_mask"]
        clean_x_dict_SPT = get_stat(clean_x, mask)
        for key in clean_x_dict_SPT:
            stat_dict[f"clean_x_woSPT_{key}"] = clean_x_dict_SPT[key]

        x_0_dict_SPT = get_stat(x_0, mask)
        for key in x_0_dict_SPT:
            stat_dict[f"x_0_woSPT_{key}"] = x_0_dict_SPT[key]

        return loss_dict, stat_dict

    def train(
            self,
            project_name: str = 'bert_diffusion',
            experiment_name: str = 'bert_emb'
    ) -> None:
        self.set_optimizer()
        self.set_scheduler()
        self.set_grad_scaler()
        self.step = 0
        self.set_valid_data_generator()
        self.file = open("log.txt", "w")
        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.model.ema_rate)

        if self.config.refresh.true:
            self.refresh_checkpoint()

            self.estimate()
            self.validate()

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
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
        for _, X in enumerate(self.train_loader):
            if self.step >= self.config.training.training_iters:
                return
            _ = next(self.train_range_iter)

            loss_dict, stat_dict = self.train_step(X)

            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

            if self.step % self.config.training.eval_freq == 0:
                self.estimate()
                self.validate()
                # self.compute_restoration_loss(suffix="train")
                # self.compute_restoration_loss(suffix="valid")

            self.train_range.set_description(
                f"loss_x_0: {loss_dict['loss_x_0'].item():0.4f}, "
                f"grad_norm: {stat_dict['grad_norm'].item():0.4f}, "
                f"accuracy: {loss_dict['accuracy'].item():0.4f}"
            )

        # torch.cuda.synchronize()

    def train_step(self, X):
        self.step += 1

        X = dict_to_cuda(X)
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            with torch.no_grad():
                clean_X = self.encoder_gen(**{"input_ids": X["input_ids"], "attention_mask": X["input_mask"]})
                cond = self.encoder_cond(**{"input_ids": X["cond_ids"], "attention_mask": X["cond_mask"]})

        loss_dict, stat_dict = self.calc_loss(clean_x=clean_X, cond=cond, X=X)

        stat_dict["grad_norm"], stat_dict["clipped_grad_norm"] = self.optimizer_step(loss_dict['total_loss'])
        stat_dict["scale_factor"] = torch.Tensor([self.grad_scaler._scale])

        if self.step % 10 == 0:
            stat_dict["weight_norm"] = torch.sqrt(
                sum([torch.sum(t.data ** 2) for t in self.score_estimator.parameters()]))

            for k, v in loss_dict.items():
                self.log_metric(k, 'train', v.item())

            for k, v in stat_dict.items():
                self.log_metric(k, 'train', v.item())

        return loss_dict, stat_dict

    def validate(self) -> None:
        prev_mode = self.ddp_score_estimator.training

        self.ddp_score_estimator.eval()
        self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        with torch.no_grad():
            for text in self.valid_loader:
                X = text
                X = dict_to_cuda(X)
                clean_X = self.encoder_gen(**{"input_ids": X["input_ids"], "attention_mask": X["input_mask"]})
                cond = self.encoder_cond(**{"input_ids": X["cond_ids"], "attention_mask": X["cond_mask"]})

                loss_dict, _ = self.calc_loss(clean_x=clean_X, cond=cond, X=X)
                for k, v in loss_dict.items():
                    if k in valid_loss:
                        valid_loss[k] += v.item() * clean_X.size(0)
                    else:
                        valid_loss[k] = torch.Tensor([v.item() * clean_X.size(0)])
                valid_count += clean_X.size(0)

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        for k, v in valid_loss.items():
            self.log_metric(k, 'valid_loader', v)

        self.switch_back_from_ema()
        self.ddp_score_estimator.train(prev_mode)

    def save_checkpoint(self, last: bool = False) -> None:
        if dist.get_rank() == 0:
            if not os.path.exists(self.checkpoints_folder):
                os.makedirs(self.checkpoints_folder)

            prefix = ''
            if self.config.checkpoints_prefix:
                prefix = self.config.checkpoints_prefix + '_'
            if last:
                prefix = prefix + 'last_'
            else:
                prefix = prefix + str(self.step) + '_'

            torch.save(
                {
                    "model": self.score_estimator.state_dict(),
                    "ema": self.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "scaler": self.grad_scaler.state_dict(),
                    "step": self.step,
                },
                os.path.join(self.checkpoints_folder, prefix + ".pth")
            )
            print(f"Save model to: {os.path.join(self.checkpoints_folder, prefix + f'model.pth')}")

    def refresh_checkpoint(self):
        if not self.config.refresh.true:
            return
        load = torch.load(f'{self.config.refresh.prefix}', map_location="cpu")

        self.ema = ExponentialMovingAverage(self.score_estimator.parameters(), self.config.model.ema_rate)
        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.switch_to_ema()

        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.grad_scaler.load_state_dict(load["scaler"])
        self.step = load["step"]
        print(f"Checkpoint refreshed {self.config.refresh.prefix}")

    @torch.no_grad()
    def generate_text(self, batch_size, cond=None, way="sde", attention_mask=None):
        cond_X, cond_mask = None, None
        with torch.no_grad():
            if cond is not None:
                cond = dict_to_cuda(cond)
                cond_X = self.encoder_cond(**{"input_ids": cond["cond"], "attention_mask": cond["cond_mask"]})
                cond_mask = cond["cond_mask"]

            if attention_mask is not None:
                attention_mask = attention_mask.cuda()

            pred_embeddings = self.pred_embeddings(
                batch_size,
                cond_X=cond_X,
                cond_mask=cond_mask,
                attention_mask=attention_mask
            )

            # pred_embeddings = normalize(pred_embeddings, dim=-1) * np.sqrt(pred_embeddings.shape[-1])
            output = self.pred_logits(pred_embeddings)
            tokens = output.argmax(dim=-1)
            text = self.tokenizer_gen.batch_decode(tokens, skip_special_tokens=True)
        return text, pred_embeddings

    @torch.no_grad()
    def pred_logits(self, pred_embeddings, input_ids=None):
        pred_embeddings = self.gen_enc_normalizer.denormalize(pred_embeddings)
        output = self.decoder(pred_embeddings)
        return output

    @torch.no_grad()
    def pred_embeddings(
            self, batch_size: int,
            cond_X=None,
            cond_mask=None,
            attention_mask=None,
    ) -> torch.Tensor:
        self.score_estimator.eval()
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            self.encoder_gen.config.hidden_size
        )

        with torch.no_grad():
            x = self.dynamic.prior_sampling(shape).to(self.device)
            x_0_self_cond = torch.zeros_like(x, dtype=x.dtype)
            eps_t = 1 / self.diff_eq_solver.dynamic.N

            if self.config.timesteps == "linear":
                timesteps = torch.linspace(self.dynamic.T, eps_t, self.dynamic.N, device=self.device)
            elif self.config.timesteps == "quad":
                deg = 2
                timesteps = torch.linspace(1, 0, self.dynamic.N, device=self.device) ** deg * (
                        self.dynamic.T - eps_t) + eps_t

            for idx in tqdm(range(self.dynamic.N)):
                t = timesteps[idx]
                next_t = timesteps[idx + 1] if idx < self.dynamic.N - 1 else eps_t  # torch.zeros_like(t)

                input_t = t * torch.ones(shape[0], device=self.device)
                next_input_t = next_t * torch.ones(shape[0], device=self.device)

                output = self.diff_eq_solver.step(
                    x_t=x, t=input_t, next_t=next_input_t,
                    cond=cond_X,
                    cond_mask=cond_mask,
                    attention_mask=attention_mask,
                    x_0_self_cond=x_0_self_cond,
                )

                x, x_mean = output["x"], output["x_mean"]
                x_0_self_cond = output["x_0"]

            pred_embeddings = x_mean

        return pred_embeddings

    @torch.no_grad()
    def estimate(self):
        self.score_estimator.eval()
        self.switch_to_ema()

        if not hasattr(self, 'metric_bloom_fn'):
            self.metric_bloom_fn = BloomMetricConditional(device=f"cuda:{dist.get_rank()}")
            self.metric_roberta_fn = RobertaMetric(device=f"cuda:{dist.get_rank()}")

        self.metric_bloom_fn.model.cuda()
        self.metric_roberta_fn.model.cuda()

        num_texts = int(self.config.validation.num_gen_texts / dist.get_world_size())
        seed = self.config.seed + dist.get_rank()
        set_seed(seed)
        metrics, joint_texts, cond_texts, gen_texts, gt_texts = estimate_model(
            self, num_texts,
            self.config.validation.batch_size,
            self.metric_bloom_fn, self.metric_roberta_fn,
        )
        joint_texts = gather_texts(joint_texts)
        cond_texts = gather_texts(cond_texts)
        gen_texts = gather_texts(gen_texts)
        gt_texts = gather_texts(gt_texts)

        metrics = reduce_metrics(metrics)
        if dist.get_rank() == 0:
            texts_path = "./generated_texts"
            print(f"Bloom metric: {metrics['Bloom metric']:0.5f}")
            print(f"Roberta metric: {metrics['Roberta metric']:0.5f}")

            text_list = []
            for i in range(len(cond_texts)):
                text_list.append(
                    {
                        "CONDITION": cond_texts[i],
                        "GEN": gen_texts[i],
                        "GT": gt_texts[i]
                    }
                )
            file_name = f"{texts_path}/{self.config.checkpoints_prefix}-{self.step}.json"
            json.dump(text_list, open(file_name, "w"), indent=4)

        self.log_metric(metric_name="bloom loss", loader_name="", value=metrics['Bloom metric'])
        self.log_metric(metric_name="roberta score", loader_name="", value=metrics['Roberta metric'])

        self.metric_bloom_fn.model.cpu()
        self.metric_roberta_fn.model.cpu()

        self.switch_back_from_ema()
        self.score_estimator.train()
        self.config.training.batch_size_per_gpu = self.config.training.batch_size // dist.get_world_size()
