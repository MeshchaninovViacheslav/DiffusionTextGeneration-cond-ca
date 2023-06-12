import torch


class Decoder(torch.nn.Module):
    def __init__(self, hidden_size=768, vocab_size=32100, layer_norm_eps=1e-12):
        super().__init__()
        self.dense = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, hidden_size),
            torch.nn.GELU(),
            torch.nn.LayerNorm(hidden_size, eps=layer_norm_eps)
        )

        self.projector = torch.nn.Linear(hidden_size, vocab_size, bias=False)
        self.bias = torch.nn.Parameter(torch.zeros(vocab_size))
        self.projector.bias = self.bias

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.dense(hidden_states)
        hidden_states = self.projector(hidden_states)
        return hidden_states
