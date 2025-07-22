"""
Learning Rate Schedulers for Semantic Segmentation Training

This module contains various learning rate scheduling strategies
including cosine annealing, polynomial decay, and warmup schedules.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
from typing import Dict, Any, Optional, List


class CosineAnnealingWithWarmup(_LRScheduler):
    """
    Cosine Annealing learning rate scheduler with linear warmup.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        T_max (int): Maximum number of iterations/epochs
        eta_min (float): Minimum learning rate
        warmup_epochs (int): Number of warmup epochs
        last_epoch (int): The index of last epoch
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, T_max: int, 
                 eta_min: float = 0, warmup_epochs: int = 0, last_epoch: int = -1):
        self.T_max = T_max
        self.eta_min = eta_min
        self.warmup_epochs = warmup_epochs
        super(CosineAnnealingWithWarmup, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Cosine annealing
            adjusted_epoch = self.last_epoch - self.warmup_epochs
            adjusted_T_max = self.T_max - self.warmup_epochs
            
            return [
                self.eta_min + (base_lr - self.eta_min) * 
                (1 + math.cos(math.pi * adjusted_epoch / adjusted_T_max)) / 2
                for base_lr in self.base_lrs
            ]


class PolynomialLR(_LRScheduler):
    """
    Polynomial learning rate decay scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        total_epochs (int): Total number of training epochs
        power (float): Power for polynomial decay
        warmup_epochs (int): Number of warmup epochs
        last_epoch (int): The index of last epoch
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, total_epochs: int, 
                 power: float = 0.9, warmup_epochs: int = 0, last_epoch: int = -1):
        self.total_epochs = total_epochs
        self.power = power
        self.warmup_epochs = warmup_epochs
        super(PolynomialLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Polynomial decay
            factor = (1 - (self.last_epoch - self.warmup_epochs) / 
                     (self.total_epochs - self.warmup_epochs)) ** self.power
            return [base_lr * factor for base_lr in self.base_lrs]


class WarmupMultiStepLR(_LRScheduler):
    """
    Multi-step learning rate scheduler with warmup.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        milestones (List[int]): List of epoch indices for learning rate decay
        gamma (float): Multiplicative factor of learning rate decay
        warmup_epochs (int): Number of warmup epochs
        last_epoch (int): The index of last epoch
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, milestones: List[int], 
                 gamma: float = 0.1, warmup_epochs: int = 0, last_epoch: int = -1):
        self.milestones = sorted(milestones)
        self.gamma = gamma
        self.warmup_epochs = warmup_epochs
        super(WarmupMultiStepLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current epoch."""
        if self.last_epoch < self.warmup_epochs:
            # Linear warmup
            warmup_factor = self.last_epoch / self.warmup_epochs
            return [base_lr * warmup_factor for base_lr in self.base_lrs]
        else:
            # Multi-step decay
            adjusted_epoch = self.last_epoch - self.warmup_epochs
            adjusted_milestones = [m - self.warmup_epochs for m in self.milestones if m > self.warmup_epochs]
            
            decay_factor = self.gamma ** len([m for m in adjusted_milestones if m <= adjusted_epoch])
            return [base_lr * decay_factor for base_lr in self.base_lrs]


class OneCycleLR(_LRScheduler):
    """
    One Cycle learning rate policy as described in "Super-Convergence".
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        max_lr (float): Maximum learning rate
        total_steps (int): Total number of training steps
        pct_start (float): Percentage of cycle spent increasing learning rate
        anneal_strategy (str): Annealing strategy ('cos' or 'linear')
        div_factor (float): Determines initial learning rate (max_lr / div_factor)
        final_div_factor (float): Determines minimum learning rate (max_lr / final_div_factor)
        last_epoch (int): The index of last epoch
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, max_lr: float, total_steps: int,
                 pct_start: float = 0.3, anneal_strategy: str = 'cos',
                 div_factor: float = 25.0, final_div_factor: float = 1e4, last_epoch: int = -1):
        self.max_lr = max_lr
        self.total_steps = total_steps
        self.pct_start = pct_start
        self.anneal_strategy = anneal_strategy
        self.div_factor = div_factor
        self.final_div_factor = final_div_factor
        
        self.initial_lr = max_lr / div_factor
        self.min_lr = max_lr / final_div_factor
        
        super(OneCycleLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        step_num = self.last_epoch
        
        if step_num <= self.pct_start * self.total_steps:
            # Increasing phase
            pct = step_num / (self.pct_start * self.total_steps)
            lr = self.initial_lr + pct * (self.max_lr - self.initial_lr)
        else:
            # Decreasing phase
            pct = (step_num - self.pct_start * self.total_steps) / ((1 - self.pct_start) * self.total_steps)
            
            if self.anneal_strategy == 'cos':
                lr = self.min_lr + (self.max_lr - self.min_lr) * (1 + math.cos(math.pi * pct)) / 2
            else:  # linear
                lr = self.max_lr - pct * (self.max_lr - self.min_lr)
        
        return [lr for _ in self.base_lrs]


class CyclicLR(_LRScheduler):
    """
    Cyclic learning rate scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        base_lr (float): Lower boundary of learning rate
        max_lr (float): Upper boundary of learning rate
        step_size_up (int): Number of training iterations in increasing half of cycle
        step_size_down (Optional[int]): Number of training iterations in decreasing half of cycle
        mode (str): One of 'triangular', 'triangular2', 'exp_range'
        gamma (float): Constant in 'exp_range' scaling function
        scale_fn (Optional[callable]): Custom scaling function
        scale_mode (str): 'cycle' or 'iterations'
        cycle_momentum (bool): Whether to cycle momentum inversely to learning rate
        base_momentum (float): Lower boundary of momentum
        max_momentum (float): Upper boundary of momentum
        last_epoch (int): The index of last epoch
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, base_lr: float, max_lr: float,
                 step_size_up: int = 2000, step_size_down: Optional[int] = None,
                 mode: str = 'triangular', gamma: float = 1.0, scale_fn: Optional[callable] = None,
                 scale_mode: str = 'cycle', cycle_momentum: bool = True,
                 base_momentum: float = 0.8, max_momentum: float = 0.9, last_epoch: int = -1):
        
        self.base_lr = base_lr
        self.max_lr = max_lr
        self.step_size_up = step_size_up
        self.step_size_down = step_size_down or step_size_up
        self.total_size = self.step_size_up + self.step_size_down
        self.mode = mode
        self.gamma = gamma
        self.scale_fn = scale_fn
        self.scale_mode = scale_mode
        self.cycle_momentum = cycle_momentum
        self.base_momentum = base_momentum
        self.max_momentum = max_momentum
        
        super(CyclicLR, self).__init__(optimizer, last_epoch)
    
    def get_lr(self) -> List[float]:
        """Compute learning rate for current step."""
        cycle = math.floor(1 + self.last_epoch / self.total_size)
        x = 1 + self.last_epoch / self.total_size - cycle
        
        if x <= self.step_size_up / self.total_size:
            scale_factor = x / (self.step_size_up / self.total_size)
        else:
            scale_factor = (x - 1) / (self.step_size_down / self.total_size) + 1
        
        # Apply scaling based on mode
        if self.scale_fn is None:
            if self.mode == 'triangular':
                lrs = [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - abs(scale_factor)))
                       for _ in self.base_lrs]
            elif self.mode == 'triangular2':
                lrs = [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - abs(scale_factor))) / (2 ** (cycle - 1))
                       for _ in self.base_lrs]
            elif self.mode == 'exp_range':
                lrs = [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - abs(scale_factor))) * (self.gamma ** self.last_epoch)
                       for _ in self.base_lrs]
        else:
            lrs = [self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - abs(scale_factor))) *
                   self.scale_fn(self.last_epoch if self.scale_mode == 'iterations' else cycle)
                   for _ in self.base_lrs]
        
        return lrs


def get_scheduler(optimizer: torch.optim.Optimizer, scheduler_config: Dict[str, Any]) -> Optional[_LRScheduler]:
    """
    Factory function to create learning rate scheduler based on configuration.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler_config (Dict[str, Any]): Scheduler configuration
        
    Returns:
        Optional[_LRScheduler]: Learning rate scheduler or None
    """
    if not scheduler_config or scheduler_config.get('type') is None:
        return None
    
    scheduler_type = scheduler_config['type'].lower()
    
    if scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', 100)
        eta_min = scheduler_config.get('min_lr', 0)
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        return CosineAnnealingWithWarmup(
            optimizer, T_max=T_max, eta_min=eta_min, warmup_epochs=warmup_epochs
        )
    
    elif scheduler_type == 'polynomial':
        total_epochs = scheduler_config.get('total_epochs', 100)
        power = scheduler_config.get('power', 0.9)
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        return PolynomialLR(
            optimizer, total_epochs=total_epochs, power=power, warmup_epochs=warmup_epochs
        )
    
    elif scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 30)
        gamma = scheduler_config.get('gamma', 0.1)
        
        return optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = scheduler_config.get('milestones', [60, 80])
        gamma = scheduler_config.get('gamma', 0.1)
        warmup_epochs = scheduler_config.get('warmup_epochs', 0)
        
        if warmup_epochs > 0:
            return WarmupMultiStepLR(
                optimizer, milestones=milestones, gamma=gamma, warmup_epochs=warmup_epochs
            )
        else:
            return optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('gamma', 0.95)
        return optim.lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'reduce_on_plateau':
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.5)
        patience = scheduler_config.get('patience', 10)
        threshold = scheduler_config.get('threshold', 1e-4)
        
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
        )
    
    elif scheduler_type == 'one_cycle':
        max_lr = scheduler_config.get('max_lr', 0.1)
        total_steps = scheduler_config.get('total_steps', 1000)
        pct_start = scheduler_config.get('pct_start', 0.3)
        anneal_strategy = scheduler_config.get('anneal_strategy', 'cos')
        div_factor = scheduler_config.get('div_factor', 25.0)
        final_div_factor = scheduler_config.get('final_div_factor', 1e4)
        
        return OneCycleLR(
            optimizer, max_lr=max_lr, total_steps=total_steps, pct_start=pct_start,
            anneal_strategy=anneal_strategy, div_factor=div_factor, final_div_factor=final_div_factor
        )
    
    elif scheduler_type == 'cyclic':
        base_lr = scheduler_config.get('base_lr', 0.001)
        max_lr = scheduler_config.get('max_lr', 0.006)
        step_size_up = scheduler_config.get('step_size_up', 2000)
        mode = scheduler_config.get('mode', 'triangular')
        gamma = scheduler_config.get('gamma', 1.0)
        
        return CyclicLR(
            optimizer, base_lr=base_lr, max_lr=max_lr, step_size_up=step_size_up,
            mode=mode, gamma=gamma
        )
    
    else:
        raise ValueError(f"Unsupported scheduler type: {scheduler_type}")


class WarmupScheduler:
    """
    Wrapper for adding warmup to any scheduler.
    
    Args:
        optimizer (torch.optim.Optimizer): Optimizer
        scheduler (_LRScheduler): Base scheduler
        warmup_epochs (int): Number of warmup epochs
        warmup_method (str): Warmup method ('linear' or 'constant')
        warmup_factor (float): Warmup factor for 'constant' method
    """
    
    def __init__(self, optimizer: torch.optim.Optimizer, scheduler: _LRScheduler,
                 warmup_epochs: int, warmup_method: str = 'linear', warmup_factor: float = 0.1):
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.warmup_epochs = warmup_epochs
        self.warmup_method = warmup_method
        self.warmup_factor = warmup_factor
        self.base_lrs = [group['lr'] for group in optimizer.param_groups]
        self.last_epoch = 0
    
    def step(self, epoch: Optional[int] = None):
        """Step the scheduler."""
        if epoch is None:
            epoch = self.last_epoch + 1
        self.last_epoch = epoch
        
        if epoch < self.warmup_epochs:
            # Warmup phase
            if self.warmup_method == 'linear':
                warmup_factor = epoch / self.warmup_epochs
            else:  # constant
                warmup_factor = self.warmup_factor
            
            for i, param_group in enumerate(self.optimizer.param_groups):
                param_group['lr'] = self.base_lrs[i] * warmup_factor
        else:
            # Normal scheduling
            self.scheduler.step(epoch - self.warmup_epochs)
    
    def state_dict(self):
        """Return state dict."""
        return {
            'scheduler': self.scheduler.state_dict(),
            'last_epoch': self.last_epoch,
            'warmup_epochs': self.warmup_epochs,
            'warmup_method': self.warmup_method,
            'warmup_factor': self.warmup_factor,
            'base_lrs': self.base_lrs
        }
    
    def load_state_dict(self, state_dict):
        """Load state dict."""
        self.scheduler.load_state_dict(state_dict['scheduler'])
        self.last_epoch = state_dict['last_epoch']
        self.warmup_epochs = state_dict['warmup_epochs']
        self.warmup_method = state_dict['warmup_method']
        self.warmup_factor = state_dict['warmup_factor']
        self.base_lrs = state_dict['base_lrs']


if __name__ == "__main__":
    # Test schedulers
    import matplotlib.pyplot as plt
    
    # Create dummy optimizer
    model = torch.nn.Linear(10, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
    
    # Test different schedulers
    schedulers = {
        'Cosine with Warmup': CosineAnnealingWithWarmup(optimizer, T_max=100, warmup_epochs=10),
        'Polynomial': PolynomialLR(optimizer, total_epochs=100, power=0.9, warmup_epochs=10),
        'One Cycle': OneCycleLR(optimizer, max_lr=0.1, total_steps=100),
    }
    
    # Plot learning rate schedules
    fig, axes = plt.subplots(1, len(schedulers), figsize=(15, 5))
    if len(schedulers) == 1:
        axes = [axes]
    
    for i, (name, scheduler) in enumerate(schedulers.items()):
        lrs = []
        # Reset optimizer
        for param_group in optimizer.param_groups:
            param_group['lr'] = 0.1
        scheduler.last_epoch = -1
        
        for epoch in range(100):
            scheduler.step()
            lrs.append(optimizer.param_groups[0]['lr'])
        
        axes[i].plot(lrs)
        axes[i].set_title(name)
        axes[i].set_xlabel('Epoch')
        axes[i].set_ylabel('Learning Rate')
        axes[i].grid(True)
    
    plt.tight_layout()
    plt.savefig('/workspace/semantic_segmentation_project/scheduler_comparison.png')
    print("Scheduler comparison plot saved!")
    
    print("Scheduler module test completed successfully!")
