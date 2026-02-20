# ema_callback.py
import copy
import torch
from pytorch_lightning.callbacks import Callback
import os

class EMACallback(Callback):
    def __init__(self, decay=0.999, update_every=1, monitor="val_mse", mode="min", eval_only_ema=True):
        self.decay = decay
        self.update_every = update_every
        self.monitor = monitor
        self.mode = mode
        self.best_score = float("inf") if mode == "min" else float("-inf")
        self.ema_model = None
        self.steps = 0
        self.eval_only_ema=eval_only_ema

    def on_train_start(self, trainer, pl_module):
        # Create shadow model
        self.ema_model = copy.deepcopy(pl_module.model)
        self.ema_model.eval()
        for param in self.ema_model.parameters():
            param.requires_grad = False

    def on_after_backward(self, trainer, pl_module):
        self.steps += 1
        if self.steps % self.update_every != 0:
            return
        with torch.no_grad():
            for ema_param, model_param in zip(self.ema_model.parameters(), pl_module.model.parameters()):
                ema_param.data.mul_(self.decay).add_(model_param.data, alpha=1.0 - self.decay)

    def on_validation_epoch_start(self, trainer, pl_module):
        if self.ema_model == None:
            return 
        if self.eval_only_ema:
            self._swap_models(pl_module)

    def on_validation_epoch_end(self, trainer, pl_module):
        if self.ema_model == None:
            return 
        metrics = trainer.callback_metrics
        current = metrics.get(self.monitor)
        if current is None:
            return  # Metric not available this epoch

        is_best = (current < self.best_score) if self.mode == "min" else (current > self.best_score)
        if is_best:
            self.best_score = current
            log_dir = trainer.logger.log_dir
            save_path = os.path.join(log_dir, f"ema_best_{self.monitor}.pt")
            torch.save(self.ema_model.state_dict(), save_path)
            print(f"[EMA] New best {self.monitor}: {current:.5f}, saved to {save_path}")

        if self.eval_only_ema:
            self._swap_models(pl_module)  # Restore original model

    def _swap_models(self, pl_module):
        for p1, p2 in zip(self.ema_model.parameters(), pl_module.model.parameters()):
            tmp = p1.data.clone()
            p1.data.copy_(p2.data)
            p2.data.copy_(tmp)
            
        