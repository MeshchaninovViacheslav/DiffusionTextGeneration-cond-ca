import os
import torch
import wandb
from copy import deepcopy
import torch.distributed as dist
from ml_collections import ConfigDict
from typing import Optional, Union, Dict
from tqdm.auto import trange
from torch.utils.data import DataLoader
from transformers import BertTokenizerFast
from tqdm import tqdm
from torch.nn.functional import cross_entropy
from timm.scheduler.cosine_lr import CosineLRScheduler

from diffusion_utils.diffusion_dynamic_sde import create_sde, create_solver
from utils.ema_model import ExponentialMovingAverage
from utils.util import dict_to_cuda, reduce_tensor
from data.dataset import RocStoryDataset

from bert_model.model_end2end import End2EndDLM


class Loss_ema_tracker:
    def __init__(self):
        self.alpha = 0.001
        self.num_step_to_fill = 100
        self._loss = 0.
        self.num_steps = 0

    def update(self, loss):
        self.num_steps += 1
        if self.num_steps < self.num_step_to_fill:
            self._loss = (self._loss * (self.num_steps - 1) + loss) / self.num_steps
        else:
            self._loss = self._loss * (1 - self.alpha) + loss * self.alpha

    @property
    def loss(self):
        return self._loss


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

        self.tokenizer = BertTokenizerFast.from_pretrained("bert-base-uncased")

        # self.load_sde()
        self.bert_config = config.bert_config
        self.model = End2EndDLM(config, self.bert_config).cuda()
        self.ddp_model = self.model
        if self.config.ddp:
            self.ddp_model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[config.local_rank],
                broadcast_buffers=False,
                find_unused_parameters=True
            )
        self.device = next(self.model.parameters()).device

        self.sde = create_sde(config=config)
        self.diff_eq_solver = create_solver(config, self.sde, ode_sampling=config.sde.ode_sampling)

        self.checkpoints_folder = config.training.checkpoints_folder
        if eval:
            self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
            self.refresh_model()
            self.model.eval()
            # self.ema = ExponentialMovingAverage(self.model.parameters(), config.model.ema_rate)
            # self.restore_parameters(self.device)
            # self.switch_to_ema()
            # self.model.eval()

        self.c4dataset = RocStoryDataset(
            tokenizer=self.tokenizer,
            max_sequence_len=self.config.data.max_sequence_len,
            split="train"
        )
        self.iter_c4dataset = iter(self.c4dataset)
        self.train_loader = None
        self.valid_loader = None

    def restore_parameters(self, device: Optional[torch.device] = None) -> None:
        checkpoints_folder: str = self.checkpoints_folder
        if device is None:
            device = torch.device('cpu')
        prefix = ''
        if self.config.checkpoints_prefix:
            prefix = self.config.checkpoints_prefix
        ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '.pth', map_location=device)["ema"]
        # ema_ckpt = torch.load(checkpoints_folder + '/' + prefix + '_ema.pth', map_location=device)
        self.ema.load_state_dict(ema_ckpt)

    def switch_to_ema(self) -> None:
        ema = self.ema
        score_model = self.model
        ema.store(score_model.parameters())
        ema.copy_to(score_model.parameters())

    def switch_back_from_ema(self) -> None:
        ema = self.ema
        score_model = self.model
        ema.restore(score_model.parameters())

    def set_optimizer(self) -> None:
        optimizer = torch.optim.AdamW(
            self.ddp_model.parameters(),
            lr=self.config.optim.lr,
            weight_decay=self.config.optim.weight_decay
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

    def set_train_data_generator(self, epoch=0) -> None:
        num_tasks = 1

        if self.train_loader is None:
            train_dataset = next(self.iter_c4dataset)
            if self.config.ddp:
                num_tasks = dist.get_world_size()
                global_rank = dist.get_rank()

                sampler_train = torch.utils.data.DistributedSampler(
                    train_dataset,
                    num_replicas=num_tasks,
                    rank=global_rank,
                    shuffle=True,
                )
            else:
                sampler_train = None

            self.train_loader = DataLoader(
                train_dataset,
                sampler=sampler_train,
                batch_size=self.config.training.batch_size // num_tasks,
                num_workers=16,
                pin_memory=True,
            )
        if self.config.ddp:
            self.train_loader.sampler.set_epoch(epoch)

        self.train_gen = iter(self.train_loader)

    def set_valid_data_generator(self) -> None:
        num_tasks = 1
        if self.valid_loader is None:
            valid_dataset = RocStoryDataset(
                tokenizer=self.tokenizer,
                max_sequence_len=self.config.data.max_sequence_len,
                split="test"
            ).dt

            if self.config.ddp:
                num_tasks = dist.get_world_size()
                sampler_valid = torch.utils.data.distributed.DistributedSampler(
                    valid_dataset,
                    shuffle=False
                )
            else:
                sampler_valid = None

            self.valid_loader = DataLoader(
                valid_dataset,
                sampler=sampler_valid,
                batch_size=self.config.validation.batch_size // num_tasks,
                num_workers=16,
                pin_memory=True,
            )

        self.valid_gen = iter(self.valid_loader)

    def log_metric(self, metric_name: str, loader_name: str, value: Union[float, torch.Tensor, wandb.Image]):
        if dist.get_rank() == 0:
            wandb.log({f'{metric_name}/{loader_name}': value}, step=self.step)

    def optimizer_step(self, loss: torch.Tensor) -> None:
        self.optimizer.zero_grad()
        loss.backward()

        if self.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                max_norm=self.grad_clip_norm
            )
        self.log_metric('lr', 'train', self.optimizer.param_groups[0]['lr'])
        self.optimizer.step()
        self.ema.update(self.model.parameters())
        self.scheduler.step_update(self.step)

    def sample_time(self, batch_size: int, eps: float = 1e-5):
        return torch.rand(batch_size) * (self.sde.T - eps) + eps

    def get_dict_to_emb(self, X: Dict) -> Dict:
        keys = [
            'input_ids',
            'token_type_ids',
            'position_ids',
            'inputs_embeds',
            'past_key_values_length'
        ]
        res = dict()
        for k in keys:
            if k in X:
                res[k] = X[k]
        return res

    def kl_loss(self, z):
        mean = torch.mean(z, dim=[0, 1])
        var = torch.var(z, dim=[0, 1])
        logvar = torch.log(var)
        return 0.5 * (torch.pow(mean, 2).sum() + var.sum() - z.shape[-1] - logvar.sum())

    def get_stat(self, z):
        mean = torch.mean(z, dim=[0, 1])
        std = torch.std(z, dim=[0, 1])
        norm = torch.norm(z, dim=[2])
        return torch.mean(mean), torch.mean(std), torch.mean(norm)

    def diffusion_loss(self, x_0, clean_x, mask):
        losses = torch.mean(torch.square(x_0 - clean_x), dim=-1)
        losses = losses * mask
        loss_x_0 = torch.sum(losses) / torch.sum(mask)
        return loss_x_0

    def reconstruction_loss(self, target, prediction_scores, mask):
        ce_losses = cross_entropy(
            input=prediction_scores.view(-1, self.bert_config.vocab_size),
            target=target.view(-1),
            reduce=False,
        )
        ce_losses = ce_losses * mask.reshape(-1)
        ce_loss = torch.sum(ce_losses) / torch.sum(mask)
        return ce_loss

    def bert_acc(self, targets, outputs, mask):
        pred_tokens = outputs.argmax(dim=-1)

        mask = deepcopy(mask)
        mask.scatter_(dim=1, index=(mask.sum(dim=1) - 1).reshape(-1, 1), src=torch.zeros_like(mask))
        mask[:, 0] = 0
        return torch.sum(mask * (targets == pred_tokens)) / torch.sum(mask)

    def calc_loss(
            self,
            X,
            eps: float = 1e-5,
    ) -> Dict[str, torch.Tensor]:
        batch_size = X["input_ids"].size(0)
        mask = X["attention_mask"]
        t = self.sample_time(batch_size, eps=eps).cuda()
        result_dict = self.ddp_model(X=X, t=t)
        x_0 = result_dict["x_0"]
        clean_x = result_dict["clean_x"]
        latent = result_dict["latent"]
        recon = result_dict["recon"]
        recon_x_0 = result_dict["recon_x_0"]

        # diffusion loss
        loss_x_0 = self.diffusion_loss(x_0=x_0, clean_x=clean_x, mask=mask)

        # denormalization loss
        losses_renorm = torch.mean(torch.square(self.model.denormalize(x_0) - latent), dim=-1)
        losses_renorm = losses_renorm * mask
        loss_z = torch.sum(losses_renorm) / torch.sum(mask)

        # reconstruction loss
        ce_loss = self.reconstruction_loss(target=X["input_ids"], prediction_scores=recon, mask=mask)

        # prediction loss
        ce_loss_pred = self.reconstruction_loss(target=X["input_ids"], prediction_scores=recon_x_0, mask=mask)

        # kl loss
        kl_loss = self.kl_loss(latent)

        loss = ce_loss + loss_x_0
        loss_dict = {
            'loss': loss,
            'total_loss': loss,
            'loss_x_0': loss_x_0,
            'loss_ce': ce_loss,
            'loss_kl': kl_loss,
            'loss_ce_pred': ce_loss_pred,
            'loss_latent': loss_z,
            'accuracy': self.bert_acc(targets=X["input_ids"], outputs=recon, mask=mask)
        }

        x_0_mean, x_0_var, x_0_norm = self.get_stat(x_0)
        clean_x_mean, clean_x_var, clean_x_norm = self.get_stat(clean_x)
        z_mean, z_var, z_norm = self.get_stat(latent)
        decoder_weight_norm = torch.norm(self.model.decoder.predictions.decoder.weight)
        decoder_transform_weight_norm = torch.norm(self.model.decoder.predictions.transform.dense.weight)

        stat_dict = {
            'x_0_mean': x_0_mean,
            'x_0_var': x_0_var,
            'x_0_norm': x_0_norm,

            'clean_x_mean': clean_x_mean,
            'clean_x_var': clean_x_var,
            'clean_x_norm': clean_x_norm,

            'z_mean': z_mean,
            'z_var': z_var,
            'z_norm': z_norm,

            'decoder_weight_norm': decoder_weight_norm,
            'decoder_transform_weight_norm': decoder_transform_weight_norm,
        }

        return loss_dict, stat_dict

    def train(
            self,
            project_name: str = 'bert_diffusion',
            experiment_name: str = 'bert_emb'
    ) -> None:
        if dist.get_rank() == 0:
            wandb.init(project=project_name, name=experiment_name, config=dict(self.config))

        self.tracker = Loss_ema_tracker()
        self.set_optimizer()
        self.set_scheduler()
        self.step = 0
        self.ema = ExponentialMovingAverage(self.model.parameters(), self.config.model.ema_rate)
        self.refresh_checkpoint()

        self.train_range = trange(self.step + 1, self.config.training.training_iters + 1)
        self.train_range_iter = iter(self.train_range)
        for epoch in range(len(self.c4dataset)):
            self.ddp_model.train()
            self.set_train_data_generator(epoch=epoch)
            self.train_epoch()

        self.model.eval()
        self.save_checkpoint(last=True)
        self.switch_to_ema()

    def train_epoch(self):
        for _, X in enumerate(self.train_loader):
            _ = next(self.train_range_iter)
            loss_dict, stat_dict = self.train_step(X)
            loss = loss_dict["loss"]
            if self.step % self.config.training.eval_freq == 0:
                self.validate()

            if self.step % self.config.training.checkpoint_freq == 0:
                self.save_checkpoint()

            self.tracker.update(loss.item())
            self.train_range.set_description(
                f"loss: {self.tracker.loss:0.5f}, "
                f"ce: {loss_dict['loss_ce'].item():0.4f}, "
                f"l_x_0: {loss_dict['loss_x_0'].item():0.4f}, "
                f"kl: {loss_dict['loss_kl'].item():0.4f}, "
                f"ce_pred: {loss_dict['loss_ce_pred'].item():0.4f}, "
                f"l_z: {loss_dict['loss_latent'].item():0.4f}, "

                f"x_0_mean: {stat_dict['x_0_mean'].item():0.4f}, "
                f"x_0_var: {stat_dict['x_0_var'].item():0.4f}, "
                f"z_mean: {stat_dict['z_mean'].item():0.4f}, "
                f"z_var: {stat_dict['z_var'].item():0.4f}, "
            )
            torch.cuda.synchronize()

    def train_step(self, X):
        self.step += 1
        X = dict_to_cuda(X)

        loss_dict, stat_dict = self.calc_loss(X=X)
        for k, v in loss_dict.items():
            self.log_metric(k, 'train', v.item())
        for k, v in stat_dict.items():
            self.log_metric(k, 'stat', v.item())
        self.optimizer_step(loss_dict['total_loss'])
        return loss_dict, stat_dict

    def validate(self) -> None:
        self.set_valid_data_generator()

        self.ddp_model.eval()
        self.model.eval()
        #self.switch_to_ema()

        valid_loss: Dict[str, torch.Tensor] = dict()
        valid_stat: Dict[str, torch.Tensor] = dict()
        valid_count = torch.Tensor([0.0])

        with torch.no_grad():
            for text in self.valid_loader:
                X = text
                X = dict_to_cuda(X)

                loss_dict, stat_dict = self.calc_loss(X=X)
                batch_size = X["input_ids"].size(0)
                for k, v in loss_dict.items():
                    if k in valid_loss:
                        valid_loss[k] += v.item() * batch_size
                    else:
                        valid_loss[k] = torch.Tensor([v.item() * batch_size])

                for k, v in stat_dict.items():
                    if k in valid_stat:
                        valid_stat[k] += v.item() * batch_size
                    else:
                        valid_stat[k] = torch.Tensor([v.item() * batch_size])

                valid_count += batch_size

        valid_count = reduce_tensor(valid_count.cuda())
        for k, v in valid_loss.items():
            valid_loss[k] = reduce_tensor(valid_loss[k].cuda())
        for k, v in valid_stat.items():
            valid_stat[k] = reduce_tensor(valid_stat[k].cuda())

        for k, v in valid_loss.items():
            valid_loss[k] = v / valid_count
        for k, v in valid_stat.items():
            valid_stat[k] = v / valid_count

        for k, v in valid_loss.items():
            self.log_metric(k, 'valid_loader', v)
        for k, v in valid_stat.items():
            self.log_metric(k, 'valid_loader', v)

        #self.switch_back_from_ema()
        self.ddp_model.train()
        self.model.train()

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
                    "model": self.model.state_dict(),
                    "ema": self.ema.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                    "scheduler": self.scheduler.state_dict(),
                    "step": self.step,
                },
                os.path.join(self.checkpoints_folder, prefix + ".pth")
            )
            print(f"Save model to: {os.path.join(self.checkpoints_folder, prefix + f'.pth')}")

    def refresh_model(self):
        path = f"{self.checkpoints_folder}/{self.config.checkpoints_prefix}.pth"
        load = torch.load(path, map_location="cpu")
        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.model.load_state_dict(load["model"])
        self.model.cuda()

    def refresh_checkpoint(self):
        if not self.config.refresh.true:
            return
        load = torch.load(f'{self.config.refresh.prefix}', map_location="cpu")
        self.ema.load_state_dict(load["ema"])
        self.ema.cuda()
        self.model.load_state_dict(load["model"])
        self.model.cuda()
        self.optimizer.load_state_dict(load["optimizer"])
        self.scheduler.load_state_dict(load["scheduler"])
        self.step = load["step"]
        print(f"Checkpoint refreshed {self.config.refresh.prefix}")

    def generate_text(self, batch_size):
        pred_embeddings = self.pred_embeddings(batch_size)
        # logits = self.model.decode(self.model.denormalize(pred_embeddings))
        logits = self.model.decode(pred_embeddings)
        tokens = logits.argmax(dim=-1)
        text = self.tokenizer.batch_decode(tokens)
        return text, pred_embeddings

    @torch.no_grad()
    def pred_embeddings(
            self, batch_size: int,
            eps: float = 1e-5,
    ) -> torch.Tensor:
        shape = (
            batch_size,
            self.config.data.max_sequence_len,
            768
        )
        with torch.no_grad():
            x = self.sde.prior_sampling(shape).to(self.device)
            eps_t = 1 / self.diff_eq_solver.sde.N
            timesteps = torch.linspace(self.sde.T, eps_t, self.sde.N, device=self.device)
            for i in tqdm(range(self.sde.N)):
                t = timesteps[i]
                vec_t = torch.ones(shape[0], device=t.device) * t
                output = self.diff_eq_solver.step(model=self.model, x_t=x, t=vec_t)
                x, x_mean = output["x"], output["x_mean"]

            pred_embeddings = x_mean

        return pred_embeddings