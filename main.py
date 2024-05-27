import os
import json
import time
from argparse import ArgumentParser

import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers.wandb import WandbLogger

from lightning import (
    add_data_specific_args,
    add_model_specific_args,
    KnowformerLightningModule, 
    TransductiveDataModule, 
    InductiveDataModule
)


dataset_type = {
    'wn18rr': 'transductive',
    'fb15k-237': 'transductive',
    'umls': 'transductive',
    'family': 'transductive',
    'nell-995': 'transductive',
    'yago3-10': 'transductive',
    'fb15k-237_v1': 'inductive',
    'fb15k-237_v2': 'inductive',
    'fb15k-237_v3': 'inductive',
    'fb15k-237_v4': 'inductive',
    'wn18rr_v1': 'inductive',
    'wn18rr_v2': 'inductive',
    'wn18rr_v3': 'inductive',
    'wn18rr_v4': 'inductive',
    'nell-995_v1': 'inductive',
    'nell-995_v2': 'inductive',
    'nell-995_v3': 'inductive',
    'nell-995_v4': 'inductive',
}

def train(args):
    pl.seed_everything(args.seed, workers=True)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':16:8'
    torch.use_deterministic_algorithms(True)
    
    dataset_name = os.path.split(args.data_path)[-1] or os.path.split(args.data_path)[-2]
    
    if dataset_type[dataset_name] == 'inductive':
        datamodule = InductiveDataModule(
            data_path=args.data_path, 
            num_workers=args.num_workers, 
            batch_size=args.batch_size, 
            test_batch_size=args.test_batch_size
        )
    else:
        datamodule = TransductiveDataModule(
            data_path=args.data_path, 
            num_workers=args.num_workers, 
            batch_size=args.batch_size, 
            test_batch_size=args.test_batch_size
        )
        
    model = KnowformerLightningModule(
        num_relation=datamodule.num_relation,
        num_layer=args.num_layer, 
        num_qk_layer=args.num_qk_layer,
        num_v_layer=args.num_v_layer,
        hidden_dim=args.hidden_dim, 
        num_heads=args.num_heads, 
        drop=args.drop,
        remove_all=args.remove_all,
        loss_fn=args.loss_fn,
        num_negative_sample=args.num_negative_sample, 
        optimizer=args.optimizer, 
        learning_rate=args.learning_rate, 
        weight_decay=args.weight_decay,
        adversarial_temperature=args.adversarial_temperature,
    )

    args.checkpoint_save_path = args.checkpoint_save_path + f'/{time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())}'
    os.makedirs(args.checkpoint_save_path, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.checkpoint_save_path, 'hparams.json'), 'w'))
    
    checkpoint_name = '-'.join((
        f"Knowformer",
        f"{dataset_name}",
        f"{args.hidden_dim}",
        f"{args.num_layer}",
        f"{args.num_negative_sample}",
        f"{args.batch_size}x{len(args.devices.split(','))}",
    ))
    checkpoint_callback = ModelCheckpoint(
        dirpath=args.checkpoint_save_path,
        filename=checkpoint_name + "_{epoch:02d}_{step}",
        every_n_epochs=1,
        monitor='valid_mrr',
        mode='max',
        save_top_k=3,
        verbose=True
    )

    wandb_logger = None 
    
    trainer = pl.Trainer(
        accelerator=args.accelerator,
        precision=args.precision,
        strategy=args.strategy,
        devices=args.devices,
        max_epochs=args.max_epochs,
        callbacks=[checkpoint_callback],
        logger=wandb_logger,
        num_sanity_val_steps=1,
        check_val_every_n_epoch=1,
    )

    trainer.fit(model, datamodule=datamodule, ckpt_path=args.resume_checkpoint_path)
    trainer.save_checkpoint(os.path.join(args.checkpoint_save_path, f"{checkpoint_name}_final.ckpt"))
    trainer.test(model, datamodule=datamodule, ckpt_path=checkpoint_callback.best_model_path)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser = pl.Trainer.add_argparse_args(parser)
    parser = add_model_specific_args(parser)
    parser = add_data_specific_args(parser)
    parser.add_argument('--seed', type=int, default=2023, help='the random seed')
    parser.add_argument('--checkpoint_save_path', type=str, default=None, help='the path to save model checkpoint')
    parser.add_argument('--resume_checkpoint_path', type=str, default=None, help='the resume model checkpoint path')
    args = parser.parse_args()
    
    train(args)
