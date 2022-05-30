# train using lightning data module and lightning module 
# and save model

import pytorch_lightning as pl

from datamodule import MyDataModule
from mykobartcls import MyKoBartCLSGenerator
from kobart import get_kobart_tokenizer

def configure_callbacks():
    checkpoint = pl.callbacks.ModelCheckpoint(
        monitor="val_acc",
        verbose=True
        )
    return [checkpoint]

if __name__ == '__main__':
    model = MyKoBartCLSGenerator(        
        model_save_path="saved/model.pt",
    )

    dm = MyDataModule(train_file="data/train.tsv",
                        test_file= "data/val.tsv",
                        tokenizer=get_kobart_tokenizer(),
                        batch_size=10,)
    
    trainer = pl.Trainer(
            gpus=1,
            distributed_backend="ddp",
            precision=16,
            # amp_backend="apex",
            amp_backend='native',
            max_epochs=2,
            callbacks=configure_callbacks()
        )

    trainer.fit(model, dm)
