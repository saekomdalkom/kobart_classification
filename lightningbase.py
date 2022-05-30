# extends Lightning Module
import pytorch_lightning as pl
import torch
from transformers.optimization import AdamW, get_cosine_schedule_with_warmup


class LightningBase(pl.LightningModule):
    def __init__(
            self,
            model_save_path: str,
            max_len: int,
            lr: float = 3e-5,
            weight_decay: float = 1e-4,
            save_step_interval: int = 1000,
            num_workers=5,
            batch_size=10,
    ) -> None:
        """constructor of LightningBase"""

        super().__init__()
        self.model_save_path = model_save_path
        self.lr = lr
        self.weight_decay = weight_decay
        self.max_len = max_len
        self.save_step_interval = save_step_interval
        self.model = None

    def configure_optimizers(self):
         # Prepare optimizer
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(
                nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(
                nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters,
                          lr=self.lr, correct_bias=False)
        # warm up lr
        num_workers = 1
        data_len = len(self.train_dataloader().dataset)
        # logging.info(f'number of workers {num_workers}, data length {data_len}')
        num_train_steps = int(data_len /10 * 2) #batch size  max epoch
        # logging.info(f'num_train_steps : {num_train_steps}')
        num_warmup_steps = int(num_train_steps * 0.1) # int(num_train_steps * self.hparams.warmup_ratio)
        # logging.info(f'num_warmup_steps : {num_warmup_steps}')
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps, num_training_steps=num_train_steps)
        lr_scheduler = {'scheduler': scheduler, 
                        'monitor': 'loss', 'interval': 'step',
                        'frequency': 1}
        return [optimizer], [lr_scheduler]

    def save_model(self) -> None:
        if (
                self.trainer.global_rank == 0
                and self.global_step % self.save_step_interval == 0
        ):
            torch.save(
                self.model.state_dict(),
                self.model_save_path + "." + str(self.global_step),
            )
            print(self.model_save_path + "." + str(self.global_step) + " has been saved.")