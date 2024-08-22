#! /usr/bin/python
# -*- coding: utf-8 -*-

"""Inverse square root scheduler with early stop."""

import math

from . import FairseqLRScheduler, register_lr_scheduler


@register_lr_scheduler('inverse_sqrt_es')
class InverseSquareRootEarlyStopSchedule(FairseqLRScheduler):
    """Inverse square root learning rate scheduler + early stop."""

    def __init__(self, args, optimizer):
        super().__init__(args, optimizer)
        if len(args.lr) > 1:
            raise ValueError(
                'Cannot use a fixed learning rate schedule with inverse_sqrt.'
                ' Consider --lr-scheduler=fixed instead.'
            )
        warmup_end_lr = args.lr[0]
        if args.warmup_init_lr < 0:
            args.warmup_init_lr = 0 if args.warmup_updates > 0 else warmup_end_lr

        # linearly warmup for the first args.warmup_updates
        self.lr_step = (warmup_end_lr - args.warmup_init_lr) / args.warmup_updates

        # then, decay prop. to the inverse square root of the update number
        self.decay_factor = warmup_end_lr * args.warmup_updates**0.5

        # initial learning rate
        self.lr = args.warmup_init_lr
        self.optimizer.set_lr(self.lr)

        # initialize early stop settings
        self.mode = 'min'
        self.threshold = 1e-4
        self.threshold_mode = 'rel'
        self.mode_worse = None  # the worse value for the chosen mode
        self.best = None
        self.num_bad_epochs = None
        self.patience = args.patience
        self._init_is_better()

        self._reset()

    @staticmethod
    def add_args(parser):
        """Add arguments to the parser for this LR scheduler."""
        # fmt: off
        parser.add_argument('--warmup-updates', default=4000, type=int, metavar='N',
                            help='warmup the learning rate linearly for the first N updates')
        parser.add_argument('--warmup-init-lr', default=-1, type=float, metavar='LR',
                            help='initial learning rate during warmup phase; default is args.lr')
        parser.add_argument('--patience', default=2, type=int, metavar='N',
                            help='number of epochs with no improvement after which the training will be stopped.')
        # fmt: on

    def _reset(self):
        """Resets num_bad_epochs counter and cooldown counter."""
        self.best = self.mode_worse
        self.cooldown_counter = 0
        self.num_bad_epochs = 0

    def is_better(self, a, best):
        if self.mode == 'min' and self.threshold_mode == 'rel':
            rel_epsilon = 1. - self.threshold
            return a < best * rel_epsilon

        elif self.mode == 'min' and self.threshold_mode == 'abs':
            return a < best - self.threshold

        elif self.mode == 'max' and self.threshold_mode == 'rel':
            rel_epsilon = self.threshold + 1.
            return a > best * rel_epsilon

        else:  # mode == 'max' and epsilon_mode == 'abs':
            return a > best + self.threshold

    def _init_is_better(self):
        if self.mode not in {'min', 'max'}:
            raise ValueError('mode ' + self.mode + ' is unknown!')
        if self.threshold_mode not in {'rel', 'abs'}:
            raise ValueError('threshold mode ' + self.threshold_mode + ' is unknown!')

        if self.mode == 'min':
            self.mode_worse = math.inf
        else:  # mode == 'max':
            self.mode_worse = -math.inf

    def state_dict(self):
        """Return the LR scheduler state dict."""
        return {
            'best': self.best,
            'num_bad_epochs': self.num_bad_epochs,
        }

    def load_state_dict(self, state_dict):
        """Load an LR scheduler state dict."""
        self.best = state_dict['best']
        self.num_bad_epochs = state_dict['num_bad_epochs']

    def step(self, epoch, val_loss=None):
        """Update the learning rate at the end of the given epoch."""

        if val_loss is None:
            return self.lr

        # convert `metrics` to float, in case it's a zero-dim Tensor
        current = float(val_loss)

        if self.is_better(current, self.best):
            self.best = current
            self.num_bad_epochs = 0
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs > self.patience:
            self.lr = -math.inf     # set LR to -INF will stop the training
            print(f'| Early stop after {self.num_bad_epochs} bad epochs.')
            self.num_bad_epochs = 0

        return self.lr

    def step_update(self, num_updates):
        """Update the learning rate after each update."""
        if num_updates < self.args.warmup_updates:
            self.lr = self.args.warmup_init_lr + num_updates*self.lr_step
        else:
            if self.lr <= 0:
                pass    # Already early stop.
            else:
                self.lr = self.decay_factor * num_updates**-0.5

        self.optimizer.set_lr(self.lr)
        return self.lr
