import torch
import wandb
import torch.distributed as dist

from util import dict_to_cuda, mse_loss


@torch.no_grad()
def compute_reconstruction_loss(diffusion, suffix="valid"):
    if dist.get_rank() != 0:
        return

    if suffix == "train":
        batch = next(iter(diffusion.train_loader))
    elif suffix == "valid":
        batch = next(iter(diffusion.valid_loader))
    
    if diffusion.config.is_conditional:
        cond = diffusion.tokenizer_cond(
            batch["text_src"],
            add_special_tokens=True,
            padding=True,
            truncation=False,
            return_tensors="pt",
            return_attention_mask=True,
            return_token_type_ids=False,
        )
        cond = dict_to_cuda(cond)
        cond_x = diffusion.encoder_cond(**{
            "input_ids": cond["input_ids"],
            "attention_mask": cond["attention_mask"]
        })
    else:
        cond, cond_x = None, None
    if cond is None:
        cond_mask = None
    else:
        cond_mask = cond.get("attention_mask", None)

    trg = diffusion.tokenizer_gen(
        batch["text_trg"],
        add_special_tokens=True,
        padding="max_length",
        max_length=diffusion.config.data.max_sequence_len,
        truncation=True,
        return_tensors="pt",
        return_attention_mask=True,
        return_token_type_ids=False,
    )
    trg = dict_to_cuda(trg)
    clean_x = diffusion.encoder_gen(**{
        "input_ids": trg["input_ids"], 
        "attention_mask": trg["attention_mask"]
    })


    batch_size = clean_x.shape[0]
    mask = None


    losses_dict = {
        "x_0 w/o selfcond": [],
        "x_0 w selfcond": [],
        "ce": [],
        "time": []
    }

    timesteps = torch.linspace(diffusion.dynamic.T, 0.001, diffusion.dynamic.N, device=diffusion.device)
    for t in timesteps:
        vec_t = t * torch.ones(batch_size, device=diffusion.device)
        x_t = diffusion.sde.marginal_forward(clean_x, vec_t)["x_t"]

        x_0_self_cond = torch.zeros_like(clean_x, dtype=clean_x.dtype)
        x_0 = diffusion.score_estimator(
            x_t=x_t, time_t=vec_t, cond=cond_x,
            attention_mask=mask, cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond
        ).detach()

        loss_x_0_wosc = mse_loss(clean_x, x_0, mask)
        x_0_self_cond = x_0

        x_0 = diffusion.score_estimator(
            x_t=x_t, time_t=vec_t, cond=cond_x,
            attention_mask=mask, cond_mask=cond_mask,
            x_0_self_cond=x_0_self_cond
        )

        loss_x_0_wsc = diffusion.mse_loss(clean_x, x_0, mask)
        loss_ce = diffusion.recon_loss(diffusion.pred_logits(pred_embeddings=x_0), trg["input_ids"], mask)

        losses_dict["x_0 w/o selfcond"].append(loss_x_0_wosc.item())
        losses_dict["x_0 w selfcond"].append(loss_x_0_wsc.item())
        losses_dict["ce"].append(loss_ce.item())
        losses_dict["time"].append(t.item())
    
    # https://docs.wandb.ai/guides/app/features/custom-charts/walkthrough
    for key in losses_dict:
        if key == "time":
            continue
        data = [[x, y] for (x, y) in zip(losses_dict["time"], losses_dict[key])]
        table = wandb.Table(data=data, columns=["time", f"loss {key}"])
        wandb.log({f"loss {key}": table})