from copy import deepcopy
from functools import partial

import einops
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import pytorch_lightning as pl

from src.model import Knowformer
from src.data import TransductiveKnowledgeGraph, InductiveKnowledgeGraph
from src.metric import MRMetric, MRRMetric, HitsMetric


class TransductiveDataModule(pl.LightningDataModule):
    def __init__(self, data_path, num_workers, batch_size, test_batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        
        self.data_object = TransductiveKnowledgeGraph(self.data_path)
        self.num_relation = self.data_object.num_relation
        
    def train_dataloader(self):
        return DataLoader(self.data_object.train_triplets.clone()[self.data_object.train_triplets[:, 1]%2==0], 
                          shuffle=True, 
                          collate_fn=self.data_object.train_collate_fn,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_object.test_triplets.clone(), 
                          shuffle=False, 
                          collate_fn=self.data_object.test_collate_fn, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.data_object.test_triplets.clone(), 
                          shuffle=False, 
                          collate_fn=self.data_object.test_collate_fn, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers)
        

class InductiveDataModule(pl.LightningDataModule):
    def __init__(self, data_path, num_workers, batch_size, test_batch_size):
        super().__init__()
        self.data_path = data_path
        self.batch_size = batch_size
        self.test_batch_size = test_batch_size
        self.num_workers = num_workers
        
        self.data_object = InductiveKnowledgeGraph(self.data_path)
        self.num_relation = self.data_object.num_relation
        
    def train_dataloader(self):
        return DataLoader(self.data_object.train_triplets.clone()[self.data_object.train_triplets[:, 1]%2==0], 
                          shuffle=True, 
                          collate_fn=self.data_object.train_collate_fn,
                          batch_size=self.batch_size, 
                          num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.data_object.test_triplets.clone(), 
                          shuffle=False, 
                          collate_fn=self.data_object.test_collate_fn, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers)
    
    def test_dataloader(self):
        return DataLoader(self.data_object.test_triplets.clone(), 
                          shuffle=False, 
                          collate_fn=self.data_object.test_collate_fn, 
                          batch_size=self.test_batch_size, 
                          num_workers=self.num_workers)


class KnowformerLightningModule(pl.LightningModule):
    def __init__(self, num_relation, num_layer, num_qk_layer, num_v_layer, hidden_dim, num_heads, drop,
                 remove_all, loss_fn, num_negative_sample, optimizer, learning_rate, weight_decay, adversarial_temperature):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = Knowformer(self.hparams.num_relation, self.hparams.num_layer, self.hparams.num_qk_layer, self.hparams.num_v_layer, 
                                self.hparams.hidden_dim, self.hparams.num_heads, self.hparams.drop)
        
        self.mr_metric_fn = MRMetric()
        self.mrr_metric_fn = MRRMetric()
        self.hits1_metric_fn = HitsMetric(topk=1)
        self.hits3_metric_fn = HitsMetric(topk=3)
        self.hits10_metric_fn = HitsMetric(topk=10)

    def remove_edge(self, batched_data):
        h_index, r_index, t_index, graph = (batched_data['h_index'], 
                                            batched_data['r_index'], 
                                            batched_data['t_index'],
                                            batched_data['graph'])
        h_index_remove = torch.cat([h_index, t_index], 0)
        r_index_remove = torch.cat([r_index, torch.where(r_index%2==0, r_index + 1, r_index - 1)], 0)
        t_index_remove = torch.cat([t_index, h_index], 0)
        
        if self.hparams.remove_all:
            # remove all edges between head and tail entities
            encode_fn = lambda x, y: x + y * graph.num_nodes
            source_hash = encode_fn(graph.edge_index[:, 0], graph.edge_index[:, 2])
            target_hash = encode_fn(h_index_remove, t_index_remove)
            mask = ~torch.isin(source_hash, target_hash)
        else:
            encode_fn = lambda x, y, z: z + (x + y * graph.num_nodes) * graph.num_nodes
            source_hash = encode_fn(graph.edge_index[:, 0], graph.edge_index[:, 1], graph.edge_index[:, 2])
            target_hash = encode_fn(h_index_remove, r_index_remove, t_index_remove)
            mask = ~torch.isin(source_hash, target_hash)

        batched_data.update({'graph_mask': mask})
        return batched_data
    
    def compute_loss(self, score, batched_data):
        positive_index = batched_data['positive_index']
        negative_index = batched_data['negative_index']
        all_index = torch.cat([positive_index, negative_index], 1)
        filter_mask = batched_data['filter_mask'].bool()
        if self.hparams.loss_fn == 'bce':
            logits = torch.gather(score, 1, all_index)
            target = torch.zeros_like(logits)
            target[:, 0] = 1
            loss = F.binary_cross_entropy_with_logits(logits, target, reduction='none')
            weights = torch.ones_like(logits)
            with torch.no_grad():
                weights[:, 1:] = F.softmax(logits[:, 1:]/self.hparams.adversarial_temperature, dim=-1)
            loss = (loss * weights).sum()
        else:
            loss = F.cross_entropy(score, positive_index.view(-1))
            
        return loss
    
    def training_step(self, batched_data):
        batch_size = batched_data['h_index'].size(0)
        num_nodes = batched_data['graph'].num_nodes
        
        batched_data['positive_index'] = batched_data['t_index'].unsqueeze(1)
        batched_data['negative_index'] = negative_sample(batched_data['filter_mask'].bool(), 
                                                         min(num_nodes, 2**self.hparams.num_negative_sample))
        
        score = self.model(self.remove_edge(batched_data))
        loss = self.compute_loss(score, batched_data)
        
        self.log('memory', torch.cuda.max_memory_allocated()/(1024**3), prog_bar=True)
        # self.log('attn-loss', attn_loss, prog_bar=True)

        return loss


    def validation_step(self, batched_data, batch_idx):
        score = self.model(batched_data)
        
        answer_score = score.gather(1, batched_data['t_index'].unsqueeze(1))
        filter_mask = batched_data['filter_mask'].bool()
        ranks = torch.sum((score >= answer_score) & (~filter_mask), dim=1) + 1

        self.mr_metric_fn.update(ranks)
        self.mrr_metric_fn.update(ranks)
        self.hits1_metric_fn.update(ranks)
        self.hits3_metric_fn.update(ranks)
        self.hits10_metric_fn.update(ranks)

    def validation_epoch_end(self, outputs):
        mr = self.mr_metric_fn.compute()
        mrr = self.mrr_metric_fn.compute()
        hits1 = self.hits1_metric_fn.compute()
        hits3 = self.hits3_metric_fn.compute()
        hits10 = self.hits10_metric_fn.compute()

        self.mr_metric_fn.reset()
        self.mrr_metric_fn.reset()
        self.hits1_metric_fn.reset()
        self.hits3_metric_fn.reset()
        self.hits10_metric_fn.reset()

        self.log('valid_mr', mr, prog_bar=True, sync_dist=True)
        self.log('valid_mrr', mrr, prog_bar=True, sync_dist=True)
        self.log('valid_hits1', hits1, prog_bar=True, sync_dist=True)
        self.log('valid_hits3', hits3, prog_bar=False, sync_dist=True)
        self.log('valid_hits10', hits10, prog_bar=True, sync_dist=True)
        
    def test_step(self, batched_data, batch_idx):
        score = self.model(batched_data)
        answer_score = score.gather(1, batched_data['t_index'].unsqueeze(1))
        filter_mask = batched_data['filter_mask'].bool()
        ranks = torch.sum((score >= answer_score) & (~filter_mask), dim=1) + 1
        
        self.mr_metric_fn.update(ranks)
        self.mrr_metric_fn.update(ranks)
        self.hits1_metric_fn.update(ranks)
        self.hits3_metric_fn.update(ranks)
        self.hits10_metric_fn.update(ranks)

    def test_epoch_end(self, outputs):
        mr = self.mr_metric_fn.compute()
        mrr = self.mrr_metric_fn.compute()
        hits1 = self.hits1_metric_fn.compute()
        hits3 = self.hits3_metric_fn.compute()
        hits10 = self.hits10_metric_fn.compute()

        self.mr_metric_fn.reset()
        self.mrr_metric_fn.reset()
        self.hits1_metric_fn.reset()
        self.hits3_metric_fn.reset()
        self.hits10_metric_fn.reset()

        self.log('test_mr', mr, prog_bar=False, sync_dist=True)
        self.log('test_mrr', mrr, prog_bar=True, sync_dist=True)
        self.log('test_hits1', hits1, prog_bar=False, sync_dist=True)
        self.log('test_hits3', hits3, prog_bar=False, sync_dist=True)
        self.log('test_hits10', hits10, prog_bar=False, sync_dist=True)
    
    def configure_optimizers(self):
        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        grouped_optimizer_parameters = [
            {
                'params': [p for n, p in self.model.named_parameters() if any([d in n for d in no_decay]) and p.requires_grad],
                'weight_decay': 0.0
            },
            {
                'params': [p for n, p in self.model.named_parameters() if not any([d in n for d in no_decay]) and p.requires_grad],
                'weight_decay': self.hparams.weight_decay
            }
        ]
        optimizer = getattr(torch.optim, self.hparams.optimizer)(
            grouped_optimizer_parameters,
            lr=self.hparams.learning_rate,
        )

        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], 0.1)
        scheduler = {
            'scheduler': scheduler, 
            'interval': 'epoch', 
            'frequency': 1
        }

        return [optimizer], [scheduler]
    
    
def add_data_specific_args(parent_args):
    parser = parent_args.add_argument_group('Data')
    parser.add_argument('--data_path', type=str, help="the path to dataset directory")
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--test_batch_size', default=16, type=int)
    return parent_args

def add_model_specific_args(parent_args):
    parser = parent_args.add_argument_group('Model')
    parser.add_argument('--num_layer', type=int, default=3, help="number of layers")
    parser.add_argument('--num_qk_layer', type=int, default=2, help="number of layers to get qk")
    parser.add_argument('--num_v_layer', type=int, default=2, help="number of layers to get v")
    parser.add_argument('--hidden_dim', type=int, default=64, help="the size of feature")
    parser.add_argument('--num_heads', type=int, default=4, help="number of heads")
    parser.add_argument('--drop', type=float, default=.1, help="dropout rate")
    parser.add_argument('--remove_all', action='store_true', help="whether or not remove all one hop edges")
    parser.add_argument('--loss_fn', type=str, default='ce', choices=['bce', 'ce'], help="loss function")
    parser.add_argument('--num_negative_sample', type=int, default=7, help="number of negative examples")
    parser.add_argument('--optimizer', type=str, default='Adam', help="the optimizer")
    parser.add_argument('--learning_rate', type=float, default=1e-4, help="the initial learning rate")
    parser.add_argument('--weight_decay', type=float, default=1e-4, help="the weight decay of optimizer")
    parser.add_argument('--adversarial_temperature', type=float, default=1.0)
    return parent_args

def positive_sample(mask):
    p = torch.ones_like(mask).float()
    p = p * mask
    pos = torch.multinomial(p, num_samples=1)
    return pos

def negative_sample(mask, num_negative_sample):
    p = torch.ones_like(mask).float()
    p = p * (~mask)
    neg = torch.multinomial(p, num_samples=num_negative_sample, replacement=True)
    return neg

# The below functions are from huggingface transformers

def _get_polynomial_decay_schedule_with_warmup_lr_lambda(
    current_step: int,
    *,
    num_warmup_steps: int,
    num_training_steps: int,
    lr_end: float,
    power: float,
    lr_init: int,
):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    elif current_step > num_training_steps:
        return lr_end / lr_init  # as LambdaLR multiplies by lr_init
    else:
        lr_range = lr_init - lr_end
        decay_steps = num_training_steps - num_warmup_steps
        pct_remaining = 1 - (current_step - num_warmup_steps) / decay_steps
        decay = lr_range * pct_remaining**power + lr_end
        return decay / lr_init  # as LambdaLR multiplies by lr_init

def get_polynomial_decay_schedule_with_warmup(
    optimizer, num_warmup_steps, num_training_steps, lr_end=1e-7, power=1.0, last_epoch=-1
):
    lr_init = optimizer.defaults["lr"]
    if not (lr_init > lr_end):
        raise ValueError(f"lr_end ({lr_end}) must be be smaller than initial lr ({lr_init})")

    lr_lambda = partial(
        _get_polynomial_decay_schedule_with_warmup_lr_lambda,
        num_warmup_steps=num_warmup_steps,
        num_training_steps=num_training_steps,
        lr_end=lr_end,
        power=power,
        lr_init=lr_init,
    )
    return LambdaLR(optimizer, lr_lambda, last_epoch)

def get_constant_schedule(optimizer, last_epoch=-1):
    return LambdaLR(optimizer, lambda x: 1.0, last_epoch=last_epoch)
