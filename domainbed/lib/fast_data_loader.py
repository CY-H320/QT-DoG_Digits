# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved

import torch


class _InfiniteSampler(torch.utils.data.Sampler):
    """Wraps another Sampler to yield an infinite stream."""

    def __init__(self, sampler):
        self.sampler = sampler

    def __iter__(self):
        while True:
            for batch in self.sampler:
                yield batch


class InfiniteDataLoader:
    def __init__(self, dataset, weights, batch_size, num_workers):
        super().__init__()

        if weights is not None:
            weights_tensor = torch.tensor(weights, dtype=torch.float)
            if weights_tensor.ndim != 1:
                raise ValueError("Weights should be a 1D sequence but given weights have shape {}".format(weights_tensor.shape))
            
            sampler = torch.utils.data.WeightedRandomSampler(weights_tensor, replacement=True, num_samples=batch_size)
        else:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=True)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler, batch_size=batch_size, drop_last=True
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
            )
        )

    def __iter__(self):
        return self  # Allows the object to be used as an iterator

    def __next__(self):
        try:
            return next(self._infinite_iterator)
        except StopIteration:
            self._infinite_iterator = iter(self._infinite_iterator)  # Reset iterator
            return next(self._infinite_iterator)


class FastDataLoader:
    """
    DataLoader wrapper with slightly improved speed by not respawning worker
    processes at every epoch.
    """

    def __init__(self, dataset, batch_size, num_workers, shuffle=False):
        super().__init__()

        if shuffle:
            sampler = torch.utils.data.RandomSampler(dataset, replacement=False)
        else:
            sampler = torch.utils.data.SequentialSampler(dataset)

        batch_sampler = torch.utils.data.BatchSampler(
            sampler,
            batch_size=batch_size,
            drop_last=False,
        )

        self._infinite_iterator = iter(
            torch.utils.data.DataLoader(
                dataset,
                num_workers=num_workers,
                batch_sampler=_InfiniteSampler(batch_sampler),
            )
        )

        self._length = len(batch_sampler)

    def __iter__(self):
        for _ in range(len(self)):
            yield next(self._infinite_iterator)

    def __len__(self):
        return self._length
