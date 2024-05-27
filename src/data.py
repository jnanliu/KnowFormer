import os
import random
import pickle
import itertools
from collections import defaultdict
from dataclasses import dataclass

import torch


@dataclass
class Graph:
    edge_index: torch.Tensor
    num_nodes: int
    num_relations: int
    
    def to(self, device):
        self.edge_index = self.edge_index.to(device)
        return self


class TransductiveKnowledgeGraph:
    def __init__(self, data_path):
        self.data_path = data_path
        
        self.entity2id = dict()
        self.id2entity = dict()
        self.relation2id = dict()
        self.id2relation = dict()
        
        with open(os.path.join(self.data_path, 'entities.txt'), 'r', encoding='utf-8') as fread:
            for line in fread:
                entity, eid = line.strip().split('\t')
                self.entity2id[entity] = int(eid)
                self.id2entity[int(eid)] = entity
                
        with open(os.path.join(self.data_path, 'relations.txt'), 'r', encoding='utf-8') as fread:
            for line in fread:
                relation, rid = line.strip().split('\t')
                self.relation2id[relation] = int(rid)
                self.id2relation[int(rid)] = relation
                
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        
        raw_train_triplets = self.read_triplets(os.path.join(self.data_path, 'train.txt'))
        raw_valid_triplets = self.read_triplets(os.path.join(self.data_path, 'valid.txt'))
        raw_test_triplets = self.read_triplets(os.path.join(self.data_path, 'test.txt'))
        
        self.train_triplets = self.encode_triplets(raw_train_triplets)
        self.valid_triplets = self.encode_triplets(raw_valid_triplets)
        self.test_triplets = self.encode_triplets(raw_test_triplets)
        
        self.train_filters = self.get_filters(self.train_triplets)
        self.valid_filters = self.get_filters(self.train_triplets + self.valid_triplets + self.test_triplets)
        self.test_filters = self.get_filters(self.train_triplets + self.valid_triplets + self.test_triplets)
        
        self.train_answers = self.get_filters(self.train_triplets)
        self.valid_answers = self.get_filters(self.valid_triplets)
        self.test_answers = self.get_filters(self.test_triplets)
        
        self.train_triplets = torch.tensor(self.train_triplets)
        self.valid_triplets = torch.tensor(self.valid_triplets)
        self.test_triplets = torch.tensor(self.test_triplets)
    
        self.edge_index = self.train_triplets.clone()
        
        self.train_graph = Graph(self.edge_index, self.num_entity, self.num_relation)
        self.valid_graph = Graph(self.edge_index, self.num_entity, self.num_relation)
        self.test_graph = Graph(self.edge_index, self.num_entity, self.num_relation)
    
    def read_triplets(self, file_name):
        triplets = list()
        with open(file_name, 'r', encoding='utf-8') as fread:
            for line in fread:
                h, r, t = line.strip().split('\t')
                triplets.append((h, r, t))
        return triplets
    
    def encode_triplets(self, raw_triplets):
        encoded_triplets = list()
        for triplet in raw_triplets:
            h, r, t = self.entity2id[triplet[0]], self.relation2id[triplet[1]], self.entity2id[triplet[2]]
            rev_r = self.relation2id['-' + triplet[1]]
            encoded_triplets.append((h, r, t))
            encoded_triplets.append((t, rev_r, h))
        return encoded_triplets
    
    def get_filters(self, triplets):
        filters = defaultdict(set)
        for h, r, t in triplets:
            filters[(h, r)].add(t)
        return filters

    def train_collate_fn(self, batch):
        batch_size = len(batch)
        for i in range(batch_size//2, batch_size):
            batch[i] = torch.flip(batch[i], [0])
            batch[i][1] = torch.where(batch[i][1]%2==0, batch[i][1] + 1, batch[i][1] - 1)
        
        filter_mask = torch.zeros(batch_size, self.train_graph.num_nodes)
        for batch_idx, query in enumerate(batch):
            filter_mask[batch_idx, list(self.train_filters[(query[0].item(), query[1].item())])] = 1

        h_index, r_index, t_index = torch.stack(batch, 0).unbind(-1)
        
        return {
            'h_index': h_index,
            'r_index': r_index,
            't_index': t_index,
            'filter_mask': filter_mask,
            'graph': self.train_graph,
        }
    
    def valid_collate_fn(self, batch):
        batch_size = len(batch)
        
        filter_mask = torch.zeros(batch_size, self.valid_graph.num_nodes)
        for batch_idx, query in enumerate(batch):
            filter_mask[batch_idx, list(self.valid_filters[(query[0].item(), query[1].item())])] = 1
        
        h_index, r_index, t_index = torch.stack(batch, 0).unbind(-1)
        return {
            'h_index': h_index,
            'r_index': r_index,
            't_index': t_index,
            'filter_mask': filter_mask,
            'graph': self.valid_graph
        }
    
    def test_collate_fn(self, batch):
        batch_size = len(batch)
        
        filter_mask = torch.zeros(batch_size, self.test_graph.num_nodes)
        for batch_idx, query in enumerate(batch):
            filter_mask[batch_idx, list(self.test_filters[(query[0].item(), query[1].item())])] = 1
        
        h_index, r_index, t_index = torch.stack(batch, 0).unbind(-1)
        return {
            'h_index': h_index,
            'r_index': r_index,
            't_index': t_index,
            'filter_mask': filter_mask,
            'graph': self.test_graph
        }    
        

class InductiveKnowledgeGraph:
    def __init__(self, data_path):
        self.data_path = data_path
        self.ind_data_path = data_path + '_ind'
        
        # train entities are disjoint of test entities
        self.entity2id = dict()
        self.id2entity = dict()
        self.ind_entity2id = dict()
        self.ind_id2entity = dict()
        # test relations are a subset of training relations
        self.relation2id = dict()
        self.id2relation = dict()
        
        with open(os.path.join(self.data_path, 'entities.txt'), 'r', encoding='utf-8') as fread:
            for line in fread:
                entity, eid = line.strip().split('\t')
                self.entity2id[entity] = int(eid)
                self.id2entity[int(eid)] = entity
        
        with open(os.path.join(self.ind_data_path, 'entities.txt'), 'r', encoding='utf-8') as fread:
            for line in fread:
                entity, eid = line.strip().split('\t')
                self.ind_entity2id[entity] = int(eid)
                self.ind_id2entity[int(eid)] = entity
        
        with open(os.path.join(self.data_path, 'relations.txt'), 'r', encoding='utf-8') as fread:
            for line in fread:
                relation, rid = line.strip().split('\t')
                self.relation2id[relation] = int(rid)
                self.id2relation[int(rid)] = relation
                
        self.num_entity = len(self.entity2id)
        self.ind_num_entity = len(self.ind_entity2id)
        self.num_relation = len(self.relation2id)
        
        raw_train_triplets = self.read_triplets(os.path.join(self.data_path, 'train.txt'))
        raw_valid_triplets = self.read_triplets(os.path.join(self.data_path, 'valid.txt'))
        raw_test_triplets = self.read_triplets(os.path.join(self.data_path, 'test.txt'))
        ind_raw_train_triplets = self.read_triplets(os.path.join(self.ind_data_path, 'train.txt'))
        ind_raw_valid_triplets = self.read_triplets(os.path.join(self.ind_data_path, 'valid.txt'))
        ind_raw_test_triplets = self.read_triplets(os.path.join(self.ind_data_path, 'test.txt'))
        
        self.train_triplets = self.encode_triplets(raw_train_triplets)
        self.valid_triplets = self.encode_triplets(raw_valid_triplets)
        self.test_triplets = self.encode_triplets(ind_raw_test_triplets, is_ind=True)
        
        # inductive train triplets construct test graph
        self.ind_train_triplets = self.encode_triplets(ind_raw_train_triplets, is_ind=True)
        self.ind_valid_triplets = self.encode_triplets(ind_raw_valid_triplets, is_ind=True)
        
        self.train_filters = self.get_filters(self.train_triplets)
        self.valid_filters = self.get_filters(self.train_triplets + self.valid_triplets)
        self.test_filters = self.get_filters(self.ind_train_triplets + self.ind_valid_triplets + self.test_triplets)
        
        self.train_answers = self.get_filters(self.train_triplets)
        self.valid_answers = self.get_filters(self.valid_triplets)
        self.test_answers = self.get_filters(self.test_triplets)
        
        self.train_triplets = torch.tensor(self.train_triplets)
        self.valid_triplets = torch.tensor(self.valid_triplets)
        self.test_triplets = torch.tensor(self.test_triplets)
        
        self.ind_train_triplets = torch.tensor(self.ind_train_triplets)
    
        self.edge_index = self.train_triplets.clone()
        self.test_edge_index = self.ind_train_triplets.clone()
        
        self.train_graph = Graph(self.edge_index, self.num_entity, self.num_relation)
        self.valid_graph = Graph(self.edge_index, self.num_entity, self.num_relation)
        self.test_graph = Graph(self.test_edge_index, self.ind_num_entity, self.num_relation)
        
    def read_triplets(self, file_name):
        triplets = list()
        with open(file_name, 'r', encoding='utf-8') as fread:
            for line in fread:
                h, r, t = line.strip().split('\t')
                triplets.append((h, r, t))
        return triplets
    
    def encode_triplets(self, raw_triplets, is_ind=False):
        encoded_triplets = list()
        for triplet in raw_triplets:
            if is_ind:
                h, r, t = self.ind_entity2id[triplet[0]], self.relation2id[triplet[1]], self.ind_entity2id[triplet[2]]
                rev_r = self.relation2id['-' + triplet[1]]
            else:
                h, r, t = self.entity2id[triplet[0]], self.relation2id[triplet[1]], self.entity2id[triplet[2]]
                rev_r = self.relation2id['-' + triplet[1]]
            encoded_triplets.append((h, r, t))
            encoded_triplets.append((t, rev_r, h))
        return encoded_triplets
    
    def get_filters(self, triplets):
        filters = defaultdict(set)
        for h, r, t in triplets:
            filters[(h, r)].add(t)
        return filters

    def train_collate_fn(self, batch):
        batch_size = len(batch)
        for i in range(batch_size//2, batch_size):
            batch[i] = torch.flip(batch[i], [0])
            batch[i][1] = torch.where(batch[i][1]%2==0, batch[i][1] + 1, batch[i][1] - 1)
        
        filter_mask = torch.zeros(batch_size, self.train_graph.num_nodes)
        for batch_idx, query in enumerate(batch):
            filter_mask[batch_idx, list(self.train_filters[(query[0].item(), query[1].item())])] = 1

        h_index, r_index, t_index = torch.stack(batch, 0).unbind(-1)
        
        return {
            'h_index': h_index,
            'r_index': r_index,
            't_index': t_index,
            'filter_mask': filter_mask,
            'graph': self.train_graph,
        }
    
    def valid_collate_fn(self, batch):
        batch_size = len(batch)
        
        filter_mask = torch.zeros(batch_size, self.valid_graph.num_nodes)
        for batch_idx, query in enumerate(batch):
            filter_mask[batch_idx, list(self.valid_filters[(query[0].item(), query[1].item())])] = 1
        
        h_index, r_index, t_index = torch.stack(batch, 0).unbind(-1)
        return {
            'h_index': h_index,
            'r_index': r_index,
            't_index': t_index,
            'filter_mask': filter_mask,
            'graph': self.valid_graph
        }
    
    def test_collate_fn(self, batch):
        batch_size = len(batch)
        
        filter_mask = torch.zeros(batch_size, self.test_graph.num_nodes)
        for batch_idx, query in enumerate(batch):
            filter_mask[batch_idx, list(self.test_filters[(query[0].item(), query[1].item())])] = 1
        
        h_index, r_index, t_index = torch.stack(batch, 0).unbind(-1)
        return {
            'h_index': h_index,
            'r_index': r_index,
            't_index': t_index,
            'filter_mask': filter_mask,
            'graph': self.test_graph
        }