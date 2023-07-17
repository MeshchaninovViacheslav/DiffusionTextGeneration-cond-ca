import pandas as pd
import torch


def calc_model_grads_norm(model: torch.nn.Module, p: float = 2):
    grads = []
    for par in model.parameters():
        if par.requires_grad and par.grad is not None:
            grads += [torch.sum(par.grad ** p)]
    return torch.pow(sum(grads), 1. / p)


import torchmetrics


class MyAccuracy(torchmetrics.Metric):
    def __init__(self, dist_sync_on_step=False):
        # call `self.add_state`for every internal state that is needed for the metrics computations
        # dist_reduce_fx indicates the function that should be used to reduce
        # state from multiple processes
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor):
        # update metric states
        assert preds.shape == target.shape

        self.correct += torch.sum(preds == target)
        self.total += target.numel()

    def compute(self):
        # compute final result
        return self.correct.float() / self.total


from lightning.pytorch.callbacks import BasePredictionWriter
import os
from itertools import chain
class Writer(BasePredictionWriter):
    def __init__(self, output_dir, write_interval):
        super().__init__(write_interval)
        self.output_dir = output_dir
        os.makedirs(self.output_dir, exist_ok=True)

    def write_on_epoch_end(self, trainer, pl_module, predictions, batch_indices):
        # this will create N (num processes) files in `output_dir` each containing
        # the predictions of it's respective rank
        predictions = [int(t) for t in chain.from_iterable(predictions)]
        # dt = pd.DataFrame(predictions, columns=["prediction"])
        # dt.to_csv(f"{self.output_dir}/SST-2.tsv", index_label="index")

        with open(f"{self.output_dir}/SST-2.tsv", 'w') as pred_fh:
            pred_fh.write("index\tprediction\n")
            split_idx = 0
            for pred in predictions:
                pred_fh.write("%d\t%s\n" % (split_idx, pred))
                split_idx += 1

