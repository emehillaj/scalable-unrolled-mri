"""Distributed sampler that can resume in the middle of an epoch."""

from typing import Iterator

import torch
from torch.utils.data import Sampler


class ResumableSampler(Sampler[int]):
    """Shards one global per-epoch order (seeded with seed + epoch) over ranks.

    Rank r yields ``order[start + r :: world_size]``. The resume position is the number of
    global samples consumed, independent of the world size. Call ``set_epoch`` every epoch
    (it resets the start), then ``set_start`` when resuming.
    """

    def __init__(
        self,
        num_samples: int,
        rank: int = 0,
        world_size: int = 1,
        seed: int = 0,
        shuffle: bool = True,
    ):
        super().__init__()
        if not 0 <= rank < world_size:
            raise ValueError(f"rank {rank} is not in [0, {world_size})")
        if num_samples < world_size:
            raise ValueError(f"{num_samples} samples cannot be split over {world_size} ranks")
        self.num_samples = int(num_samples)
        self.rank = rank
        self.world_size = world_size
        self.seed = seed
        self.shuffle = shuffle
        self.epoch = 0
        self.start = 0

    def set_epoch(self, epoch: int) -> None:
        """Select the order of ``epoch`` and restart from its beginning."""
        self.epoch = int(epoch)
        self.start = 0

    def set_start(self, samples_consumed: int) -> None:
        """Skip the first ``samples_consumed`` global samples of the epoch."""
        if not 0 <= samples_consumed <= self.num_samples:
            raise ValueError(f"start {samples_consumed} is not in [0, {self.num_samples}]")
        self.start = int(samples_consumed)

    def global_order(self) -> list:
        """The full order of the current epoch."""
        if not self.shuffle:
            return list(range(self.num_samples))
        generator = torch.Generator()
        generator.manual_seed(self.seed + self.epoch)
        return torch.randperm(self.num_samples, generator=generator).tolist()

    def __len__(self) -> int:
        return (self.num_samples - self.start) // self.world_size

    def __iter__(self) -> Iterator[int]:
        order = self.global_order()
        end = self.start + len(self) * self.world_size
        return iter(order[self.start + self.rank : end : self.world_size])
