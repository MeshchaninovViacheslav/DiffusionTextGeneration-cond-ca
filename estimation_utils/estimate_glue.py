import torch
from utils.util import dict_to_cuda


@torch.no_grad()
def estimate_sst2(diffusion):
    diffusion.set_valid_data_generator()

    num_right = 0.
    num = 0.

    for X in diffusion.valid_loader:
        X = dict_to_cuda(X)
        cond_X = diffusion.sampler_emb({"input_ids": X["cond_ids"], "attention_mask": X["cond_mask"]})
        pred_embeddings = diffusion.pred_embeddings(cond_X.shape[0], cond_X=cond_X, cond_mask=X["cond_mask"])
        output = diffusion.pred_logits(pred_embeddings)

        positive_ind = 2748#3893
        negative_ind = 2053#4997

        positive_proba = output[:, 1, positive_ind]
        negative_proba = output[:, 1, negative_ind]

        label = torch.where(
            negative_proba < positive_proba,
            positive_ind,
            negative_ind
        )
        target = X["input_ids"][:, 1]

        num_right += torch.sum((label == target) * 1.).item()
        num += target.shape[0]
    return num_right, num

