import logging
from typing import Callable, Optional

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.results import Result


class Trainer:
    def __init__(self, model: nn.Module,
                 device: torch.device,
                 *,
                 criterion: nn.Module,
                 num_epoch: Optional[int] = 5,
                 verbose: bool = False,
                 optimizer: Optional[Optimizer] = None,
                 train_dataloader: Optional[DataLoader] = None,
                 test_dataloader: Optional[DataLoader] = None,
                 eval_metric: Optional[Callable] = None,
                 ) -> None:
        self._device = device

        self._model = model.to(self._device)
        self._criterion = criterion
        self._optimizer = optimizer

        self._eval_metric = eval_metric

        self._train_dataloader = train_dataloader
        self._test_dataloader = test_dataloader

        self._num_epoch = num_epoch
        self._verbose = verbose

        self._best_score = .0
        self._best_model = self._model

    def train(self) -> None:

        for epoch in range(self._num_epoch):
            self._model.train()
            train_losses = []

            for batch in tqdm(self._train_dataloader, desc='Train'):
                input_ids = batch['input_ids'].to(self._device)
                labels = batch['label'].to(self._device)

                self._optimizer.zero_grad()
                prediction = self._model(input_ids)
                loss = self._criterion(prediction, labels)
                loss.backward()
                self._optimizer.step()

                train_losses.append(loss.item())

            if self._test_dataloader:
                eval_result, _ = self.eval()
                if self._best_score < eval_result.metric:
                    self._best_score = eval_result.metric
                    self._best_model = self._model

            if self._verbose:
                logging.warning('Epoch [%i/%i]:\ttrain_loss %.2f\tval_loss %.2f\tscore %.2f',
                                epoch, self._num_epoch, np.mean(train_losses), eval_result.loss, eval_result.metric)
        logging.warning('Training complete: best score %.2f', self._best_score)

    @property
    def best_model(self) -> nn.Module:
        return self._best_model

    @property
    def best_score(self) -> float:
        return self._best_score

    @torch.inference_mode()
    def eval(self) -> tuple[Result, Tensor]:
        self._model.eval()

        predictions = []
        labels = []
        eval_losses = []

        for batch in self._test_dataloader:
            input_ids = batch['input_ids'].to(self._device)
            logits = self._model(input_ids)

            if 'label' in batch:
                label = batch['label'].to(self._device)
                loss = self._criterion(logits, label)
                labels.append(label)
                eval_losses.append(loss.item())

            predictions.append(logits.argmax(dim=1))
    
        pred = torch.cat(predictions)
        if labels:
            targets = torch.cat(labels)
            metric = self._eval_metric(pred, targets)
            return Result(np.mean(eval_losses), metric), pred
        return Result(.0, .0), pred
