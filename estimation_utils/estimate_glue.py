import torch
from utils.util import dict_to_cuda


@torch.no_grad()
def estimate_sst2(diffusion):
    diffusion.set_valid_data_generator()

    num_right = 0.
    num = 0.
    n_ans = 1
    positive_ind = 3893
    negative_ind = 4997

    for X in diffusion.valid_loader:
        labels = []
        for i in range(n_ans):
            X = dict_to_cuda(X)
            cond_X = diffusion.sampler_emb({"input_ids": X["cond_ids"], "attention_mask": X["cond_mask"]})

            pred_embeddings = diffusion.pred_embeddings(cond_X.shape[0], cond_X=cond_X, cond_mask=X["cond_mask"])
            output = diffusion.pred_logits(pred_embeddings)

            positive_proba = output[:, 1, positive_ind]
            negative_proba = output[:, 1, negative_ind]

            label = torch.where(
                negative_proba < positive_proba,
                1.,
                0.
            )
            labels.append(label)

        labels = torch.stack(labels, dim=1)
        labels = torch.where(
            torch.mean(labels, dim=1) > 0.5,
            positive_ind,
            negative_ind
        )

        target = X["input_ids"][:, 1]

        num_right += torch.sum((labels == target) * 1.).item()
        num += target.shape[0]
    return num_right, num
