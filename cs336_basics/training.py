import torch
import torch.nn as nn
import numpy as np
import math
import os
from typing import Union, BinaryIO, IO


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    original_shape = logits.shape[:-1]
    vocab_size = logits.shape[-1]
    
    logits_flat = logits.reshape(-1, vocab_size)
    targets_flat = targets.reshape(-1)
    
    logits_max = torch.max(logits_flat, dim=-1, keepdim=True)[0]
    logits_shifted = logits_flat - logits_max
    
    log_sum_exp = torch.log(torch.sum(torch.exp(logits_shifted), dim=-1, keepdim=True))
    log_softmax = logits_shifted - log_sum_exp
    
    batch_size = logits_flat.shape[0]
    indices = torch.arange(batch_size, device=logits.device)
    log_probs = log_softmax[indices, targets_flat]
    
    loss = -log_probs.mean()
    
    return loss


class AdamW(torch.optim.Optimizer):
    def __init__(
        self,
        params,
        lr: float = 1e-3,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 0.01
    ):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta1: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta2: {betas[1]}")
        if eps < 0.0:
            raise ValueError(f"Invalid epsilon: {eps}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay: {weight_decay}")
        
        defaults = dict(
            lr=lr,
            betas=betas,
            eps=eps,
            weight_decay=weight_decay
        )
        super().__init__(params, defaults)
    
    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            eps = group['eps']
            weight_decay = group['weight_decay']
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad.data
                
                state = self.state[p]
                
                if len(state) == 0:
                    state['t'] = 0
                    state['m'] = torch.zeros_like(p.data)
                    state['v'] = torch.zeros_like(p.data)
                
                m = state['m']
                v = state['v']
                state['t'] += 1
                t = state['t']
                
                m.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                bias_correction1 = 1 - beta1 ** t
                bias_correction2 = 1 - beta2 ** t
                adjusted_lr = lr * math.sqrt(bias_correction2) / bias_correction1
                
                p.data.addcdiv_(m, v.sqrt().add_(eps), value=-adjusted_lr)
                
                p.data.mul_(1 - lr * weight_decay)
        
        return loss


def cosine_learning_rate_schedule(
    it: int,
    max_learning_rate: float,
    min_learning_rate: float,
    warmup_iters: int,
    cosine_cycle_iters: int
) -> float:
    if it < warmup_iters:
        return (it / warmup_iters) * max_learning_rate
    
    elif it <= cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        return min_learning_rate + 0.5 * (max_learning_rate - min_learning_rate) * (
            1 + math.cos(progress * math.pi)
        )
    
    else:
        return min_learning_rate


def gradient_clipping(parameters, max_norm: float, eps: float = 1e-6):
    total_norm = 0.0
    for p in parameters:
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    
    total_norm = math.sqrt(total_norm)
    
    if total_norm > max_norm:
        clip_coef = max_norm / (total_norm + eps)
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)


def get_batch(
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str
) -> tuple[torch.Tensor, torch.Tensor]:
    max_start_idx = len(dataset) - context_length
    
    if max_start_idx <= 0:
        raise ValueError(
            f"数据集太小：长度 {len(dataset)}，需要至少 {context_length + 1}"
        )
    
    start_indices = np.random.randint(0, max_start_idx, size=batch_size)
    
    x = np.array([dataset[i:i+context_length] for i in start_indices])
    y = np.array([dataset[i+1:i+context_length+1] for i in start_indices])
    
    x = torch.from_numpy(x).long().to(device)
    y = torch.from_numpy(y).long().to(device)
    
    return x, y


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    iteration: int,
    out: Union[str, os.PathLike, BinaryIO, IO[bytes]]
):
    checkpoint = {
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'iteration': iteration
    }
    
    torch.save(checkpoint, out)


def load_checkpoint(
    src: Union[str, os.PathLike, BinaryIO, IO[bytes]],
    model: nn.Module,
    optimizer: torch.optim.Optimizer
) -> int:
    checkpoint = torch.load(src, map_location='cpu')
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['iteration']


def compute_perplexity(losses: list[float]) -> float:
    avg_loss = sum(losses) / len(losses)
    return math.exp(avg_loss)


def train_step(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x: torch.Tensor,
    y: torch.Tensor,
    max_grad_norm: float = 1.0
) -> float:
    model.train()
    
    logits = model(x)
    
    loss = cross_entropy(logits, y)
    
    optimizer.zero_grad()
    loss.backward()
    
    gradient_clipping(model.parameters(), max_grad_norm)
    
    optimizer.step()
    
    return loss.item()


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataset: np.ndarray,
    batch_size: int,
    context_length: int,
    device: str,
    num_batches: int = 100
) -> float:
    model.eval()
    
    total_loss = 0.0
    for _ in range(num_batches):
        x, y = get_batch(dataset, batch_size, context_length, device)
        logits = model(x)
        loss = cross_entropy(logits, y)
        total_loss += loss.item()
    
    return total_loss / num_batches