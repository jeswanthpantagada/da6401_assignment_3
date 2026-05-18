import math
from typing import List

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import LRScheduler

import matplotlib.pyplot as plt


class NoamScheduler(LRScheduler):
    """
    Implements the Noam Learning Rate Scheduler from
    'Attention Is All You Need'.

    Formula:
        lrate = d_model^(-0.5) *
                min(step_num^(-0.5),
                    step_num * warmup_steps^(-1.5))
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_epoch: int = -1,
    ) -> None:

        self.d_model = d_model
        self.warmup_steps = warmup_steps

        super().__init__(optimizer, last_epoch)

    def _get_lr_scale(self, step: int) -> float:
        """
        Computes the scaling factor for the current step.
        """

        step = max(step, 1)

        scale = (
            (self.d_model ** -0.5)
            * min(step ** -0.5, step * (self.warmup_steps ** -1.5))
        )

        return scale

    def get_lr(self) -> List[float]:
        """
        Returns the learning rate for each parameter group.
        """

        step = self.last_epoch + 1
        scale = self._get_lr_scale(step)

        return [base_lr * scale for base_lr in self.base_lrs]


def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
    base_lr: float = 1.0,
) -> List[float]:
    """
    Generates LR history for visualization/testing.
    """

    dummy_model = torch.nn.Linear(1, 1)

    optimizer = optim.Adam(
        dummy_model.parameters(),
        lr=base_lr,
    )

    scheduler = NoamScheduler(
        optimizer=optimizer,
        d_model=d_model,
        warmup_steps=warmup_steps,
    )

    lr_history = []

    for _ in range(total_steps):
        optimizer.step()
        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]
        lr_history.append(current_lr)

    return lr_history


def plot_lr_schedule(
    d_model: int = 512,
    warmup_steps: int = 4000,
    total_steps: int = 20000,
) -> None:
    """
    Plots the Noam learning rate schedule.
    """

    lr_history = get_lr_history(
        d_model=d_model,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    plt.figure(figsize=(10, 5))

    plt.plot(lr_history)

    plt.axvline(
        warmup_steps,
        linestyle="--",
        label=f"Warmup Steps = {warmup_steps}",
    )

    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Noam Learning Rate Schedule")
    plt.legend()

    plt.tight_layout()
    plt.show()


def verify_scheduler_properties(
    d_model: int = 512,
    warmup_steps: int = 4000,
    total_steps: int = 10000,
) -> None:
    """
    Verifies important properties expected by assignment tests.
    """

    lrs = get_lr_history(
        d_model=d_model,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )

    peak_step = lrs.index(max(lrs)) + 1

    print("=" * 50)
    print("Noam Scheduler Verification")
    print("=" * 50)

    print(f"Peak LR occurs near warmup step:")
    print(f"Peak Step = {peak_step}")
    print(f"Warmup Steps = {warmup_steps}")

    increasing = all(
        lrs[i] <= lrs[i + 1]
        for i in range(warmup_steps - 2)
    )

    decreasing = all(
        lrs[i] >= lrs[i + 1]
        for i in range(warmup_steps, total_steps - 1)
    )

    print(f"\nMonotonic increase during warmup: {increasing}")
    print(f"Monotonic decrease after warmup: {decreasing}")

    theoretical_peak = (
        (d_model ** -0.5)
        * (warmup_steps ** -0.5)
    )

    print(f"\nTheoretical peak LR: {theoretical_peak:.10f}")
    print(f"Observed peak LR:    {max(lrs):.10f}")

    step1_lr = (
        (d_model ** -0.5)
        * min(
            1 ** -0.5,
            1 * (warmup_steps ** -1.5),
        )
    )

    print(f"\nExpected LR at step 1: {step1_lr:.10f}")
    print(f"Observed LR at step 1: {lrs[0]:.10f}")

    print("=" * 50)


if __name__ == "__main__":

    D_MODEL = 512
    WARMUP_STEPS = 4000
    TOTAL_STEPS = 20000

    verify_scheduler_properties(
        d_model=D_MODEL,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_STEPS,
    )

    plot_lr_schedule(
        d_model=D_MODEL,
        warmup_steps=WARMUP_STEPS,
        total_steps=TOTAL_STEPS,
    )
