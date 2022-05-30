from logging import log
import pytorch_lightning as pl
from lightningbase import LightningBase
from transformers import BartForSequenceClassification
from kobart import get_pytorch_kobart_model
import torch
from torchmetrics.functional import accuracy 


class MyKoBartCLSGenerator(LightningBase):
    def __init__(
        self,
        model_save_path: str,
        max_len: int = 512,
        lr: float = 5e-5,
        weight_decay: float = 1e-4,
        save_step_interval: int = 1000,
    ) -> None:
        super(MyKoBartCLSGenerator, self).__init__(
            model_save_path=model_save_path,
            max_len=max_len,
            lr=lr,
            weight_decay=weight_decay,
            save_step_interval=save_step_interval,
        )

        self.model = BartForSequenceClassification.from_pretrained(get_pytorch_kobart_model(), num_labels=6)
        self.model.train()
        self.metric_acc = pl.metrics.classification.Accuracy()

    def forward(self, input_ids, attention_mask, labels=None):
        # print(input_ids, attention_mask, labels)
        return self.model(input_ids=input_ids, attention_mask=attention_mask, labels=labels, return_dict=True)

    def training_step(self, batch, batch_idx):
        outs = self(batch['input_ids'], batch['attention_mask'], batch['labels'])
        loss = outs.loss
        self.log('train_loss', loss, prog_bar=True)
        return loss

        # logits = torch.nn.functional.log_softmax(pred.logits, dim=1)
        # loss = torch.nn.functional.nll_loss(logits, labels)

        # # validation metrics
        # preds = torch.argmax(logits, dim=1)
        # acc = accuracy(preds, labels)
        # self.log('train_loss', loss, on_step=True, on_epoch=True, logger=True)
        # self.log('train_acc', acc, on_step=True, on_epoch=True, logger=True)

        # self.save_model()
        # return loss

    def validation_step(self, batch, batch_idx):
        # # print("asdasd", batch)
        # pred = self(batch['input_ids'], batch['attention_mask'])
        # labels = batch['labels']

        # logits = torch.nn.functional.log_softmax(pred.logits, dim=1)
        # loss = torch.nn.functional.nll_loss(logits, labels)

        # # validation metrics
        # preds = torch.argmax(pred.logits, dim=1)
        # acc = accuracy(preds, labels)
        # self.log('val_loss', loss, prog_bar=True)
        # self.log('val_acc', acc, prog_bar=True)

        # return loss


        pred = self(batch['input_ids'], batch['attention_mask'])
        labels = batch['labels']
        accuracy = self.metric_acc(torch.nn.functional.softmax(pred.logits, dim=1), labels)
        self.log('accuracy', accuracy)
        result = {'accuracy': accuracy}
        # Checkpoint model based on validation loss
        return result

    def validation_epoch_end(self, outputs):
        # losses = []
        # for loss in outputs:
        #     losses.append(loss)
        # self.log('val_loss', torch.stack(losses).mean(), prog_bar=True)
        val_acc = torch.stack([i['accuracy'] for i in outputs]).mean()
        self.log('val_acc', val_acc, prog_bar=True)