import math
from typing import List

import torch
import torch.optim as optim
import matplotlib.pyplot as plt


class NoamScheduler:
    """
    Noam learning-rate schedule from the Transformer paper.

    lrate = d_model^(-0.5) * min(step^(-0.5), step * warmup_steps^(-1.5))
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        d_model: int,
        warmup_steps: int,
        last_step: int = 0,
    ) -> None:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive")

        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = last_step
        self._last_lr = [group["lr"] for group in optimizer.param_groups]

    def _rate(self, step: int) -> float:
        step = max(step, 1)
        return (self.d_model ** -0.5) * min(
            step ** -0.5,
            step * (self.warmup_steps ** -1.5),
        )

    def step(self) -> float:
        self.step_num += 1
        lr = self._rate(self.step_num)

        for group in self.optimizer.param_groups:
            group["lr"] = lr

        self._last_lr = [lr for _ in self.optimizer.param_groups]
        return lr

    def get_last_lr(self) -> List[float]:
        return self._last_lr

    def state_dict(self) -> dict:
        return {
            "d_model": self.d_model,
            "warmup_steps": self.warmup_steps,
            "step_num": self.step_num,
            "last_lr": self._last_lr,
        }

    def load_state_dict(self, state_dict: dict) -> None:
        self.d_model = state_dict["d_model"]
        self.warmup_steps = state_dict["warmup_steps"]
        self.step_num = state_dict["step_num"]
        self._last_lr = state_dict.get("last_lr", self._last_lr)

        if self.step_num > 0:
            lr = self._rate(self.step_num)
            for group in self.optimizer.param_groups:
                group["lr"] = lr
            self._last_lr = [lr for _ in self.optimizer.param_groups]


def get_lr_history(
    d_model: int,
    warmup_steps: int,
    total_steps: int,
) -> List[float]:
    dummy_model = torch.nn.Linear(1, 1)
    optimizer = optim.Adam(dummy_model.parameters(), lr=1.0)
    scheduler = NoamScheduler(optimizer, d_model=d_model, warmup_steps=warmup_steps)

    history = []
    for _ in range(total_steps):
        history.append(scheduler.step())

    return history


def plot_lr_schedule(
    d_model: int = 512,
    warmup_steps: int = 4000,
    total_steps: int = 20000,
) -> None:
    lrs = get_lr_history(d_model, warmup_steps, total_steps)

    plt.figure(figsize=(10, 5))
    plt.plot(lrs)
    plt.axvline(warmup_steps, linestyle="--", label=f"Warmup Steps = {warmup_steps}")
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
    lrs = get_lr_history(
        d_model=d_model,
        warmup_steps=warmup_steps,
        total_steps=total_steps,
    )
    peak_step = lrs.index(max(lrs)) + 1

    print("=" * 50)
    print("Noam Scheduler Verification")
    print("=" * 50)
    print(f"Peak Step = {peak_step}")
    print(f"Warmup Steps = {warmup_steps}")

    increasing = all(lrs[i] <= lrs[i + 1] for i in range(min(warmup_steps - 1, len(lrs) - 1)))
    decreasing = all(lrs[i] >= lrs[i + 1] for i in range(warmup_steps, len(lrs) - 1))

    print(f"\nMonotonic increase during warmup: {increasing}")
    print(f"Monotonic decrease after warmup: {decreasing}")

    theoretical_peak = (d_model ** -0.5) * (warmup_steps ** -0.5)
    print(f"\nTheoretical peak LR: {theoretical_peak:.10f}")
    print(f"Observed peak LR:    {max(lrs):.10f}")

    step1_lr = (d_model ** -0.5) * min(1 ** -0.5, 1 * (warmup_steps ** -1.5))
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
