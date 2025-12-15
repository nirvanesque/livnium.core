"""
Training Schedules: Learning Rate and Lambda Schedules

Schedules for learning rates, lambda values, and other training hyperparameters.
These are training conveniences, not kernel laws.
"""

from typing import Callable, Optional


class LearningRateSchedule:
    """
    Learning rate schedule.
    
    Provides learning rate at each step/epoch.
    """
    
    def __init__(self, initial_lr: float, schedule_fn: Optional[Callable[[int], float]] = None):
        """
        Initialize learning rate schedule.
        
        Args:
            initial_lr: Initial learning rate
            schedule_fn: Optional function(step) -> lr
        """
        self.initial_lr = initial_lr
        self.schedule_fn = schedule_fn or (lambda step: initial_lr)
    
    def __call__(self, step: int) -> float:
        """
        Get learning rate for given step.
        
        Args:
            step: Training step
            
        Returns:
            Learning rate
        """
        return self.schedule_fn(step)


def linear_warmup_schedule(initial_lr: float, warmup_steps: int) -> Callable[[int], float]:
    """
    Create linear warmup schedule.
    
    Args:
        initial_lr: Target learning rate after warmup
        warmup_steps: Number of warmup steps
        
    Returns:
        Schedule function
    """
    def schedule(step: int) -> float:
        if step < warmup_steps:
            return initial_lr * (step / warmup_steps)
        return initial_lr
    return schedule


def cosine_decay_schedule(
    initial_lr: float,
    min_lr: float,
    total_steps: int
) -> Callable[[int], float]:
    """
    Create cosine decay schedule.
    
    Args:
        initial_lr: Initial learning rate
        min_lr: Minimum learning rate
        total_steps: Total training steps
        
    Returns:
        Schedule function
    """
    import math
    
    def schedule(step: int) -> float:
        if step >= total_steps:
            return min_lr
        progress = step / total_steps
        return min_lr + (initial_lr - min_lr) * 0.5 * (1 + math.cos(math.pi * progress))
    return schedule


class LambdaSchedule:
    """
    Lambda schedule for pairwise distillation (from ecw-BT).
    
    Lambda controls fusion weight in pairwise distillation.
    """
    
    def __init__(self, schedule: dict[int, float]):
        """
        Initialize lambda schedule.
        
        Args:
            schedule: Dict mapping epoch -> lambda value
                     Example: {1: 0.05, 2: 0.10, 3: 0.15}
        """
        self.schedule = schedule
        self.default = max(schedule.values()) if schedule else 0.15
    
    def __call__(self, epoch: int) -> float:
        """
        Get lambda for given epoch.
        
        Args:
            epoch: Training epoch (1-indexed)
            
        Returns:
            Lambda value
        """
        # Find the highest epoch <= current epoch
        for e in sorted(self.schedule.keys(), reverse=True):
            if epoch >= e:
                return self.schedule[e]
        return self.default


def default_lambda_schedule() -> LambdaSchedule:
    """
    Default lambda schedule from ecw-BT.
    
    epoch1: lambda = 0.05
    epoch2: lambda = 0.10
    epoch3+: lambda = 0.15
    """
    return LambdaSchedule({
        1: 0.05,
        2: 0.10,
        3: 0.15,
    })

