import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn

import pytorch_lightning as pl
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger

from config.args import args
from data.dataset import WN18Dataset
from model.model import GCNLayersRandomLeaps, ConvKB
from util.util import read_id, generate_adj, calc_trans_e_mse_loss, calc_tran_e_margin_ranking_loss, calc_conv_loss


class PreTrainGCN(pl.LightningModule):
    def __init__(self, dataset_name: str = 'WN18RR'):
        super(PreTrainGCN, self).__init__()
        self.data_path = os.path.join('data', dataset_name)
        self.model = GCNLayersRandomLeaps(
            n_feat=args.embedding_dim,
            dropout=args.dropout
        ).to(device=self.device)

    def forward(self, x, adj):
        return self.model(x, adj)

    def prepare_data(self):
        # entity / relation 2 id
        self.entity2id = read_id(os.path.join(self.data_path, 'entity2id.txt'))
        self.relation2id = read_id(os.path.join(self.data_path, 'relation2id.txt'))
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        # train / validation / test dataset
        self.train_dataset = WN18Dataset(data_path=self.data_path, dataset_name='train', entity2id=self.entity2id, relation2id=self.relation2id)
        self.val_dataset = WN18Dataset(data_path=self.data_path, dataset_name='valid', entity2id=self.entity2id, relation2id=self.relation2id)
        self.test_dataset = WN18Dataset(data_path=self.data_path, dataset_name='test', entity2id=self.entity2id, relation2id=self.relation2id)
        # init entity / relation embeddings
        self.entity_embeddings = nn.Embedding(self.num_entity, args.embedding_dim).to(device=self.device)
        self.relation_embeddings = nn.Embedding(self.num_relation, args.embedding_dim).to(device=self.device)
        # init train / validation / test adjacency matrix
        self.train_adj = generate_adj(data=self.train_dataset.data, num_entity=self.num_entity)
        self.val_adj = generate_adj(data=self.val_dataset.data, num_entity=self.num_entity)
        self.test_adj = generate_adj(data=self.test_dataset.data, num_entity=self.num_entity)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.factor)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        h, r, t, v = batch
        self(self.entity_embeddings(torch.arange(self.num_entity).to(device=self.device)), self.train_adj)

        # mse_loss = calc_trans_e_mse_loss(
        #     entity_embeddings=self.entity_embeddings,
        #     relation_embeddings=self.relation_embeddings,
        #     triples=(self.entity_embeddings(h), self.relation_embeddings(r), self.entity_embeddings(h)),
        #     batch_size=len(r),
        #     embedding_dim=args.embedding_dim,
        # ).to(device=self.device)
        margin_ranking_loss = calc_tran_e_margin_ranking_loss(
            triples=(
                self.entity_embeddings(h),
                self.relation_embeddings(r),
                self.entity_embeddings(h)
            ),
            values=v,
            margin=args.margin,
        ).to(device=self.device)

        if hasattr(torch.cuda, 'empty_cache'):
            torch.cuda.empty_cache()

        self.log('train_loss', margin_ranking_loss, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': margin_ranking_loss}

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


class TrainConvKB(pl.LightningModule):
    def __init__(self, dataset_name: str = 'WN18RR'):
        super(TrainConvKB, self).__init__()
        self.data_path = os.path.join('data', dataset_name)
        self.model = ConvKB(
            input_dim=args.embedding_dim,
            in_channels=1,
            out_channels=3,
            drop_prob=args.dropout,
        ).to(device=self.device)

    def forward(self, x):
        return self.model(x)

    def prepare_data(self):
        # entity / relation 2 id
        self.entity2id = read_id(os.path.join(self.data_path, 'entity2id.txt'))
        self.relation2id = read_id(os.path.join(self.data_path, 'relation2id.txt'))
        self.num_entity = len(self.entity2id)
        self.num_relation = len(self.relation2id)
        # train / validation / test dataset
        self.train_dataset = WN18Dataset(data_path=self.data_path, dataset_name='train', entity2id=self.entity2id, relation2id=self.relation2id)
        self.val_dataset = WN18Dataset(data_path=self.data_path, dataset_name='valid', entity2id=self.entity2id, relation2id=self.relation2id)
        self.test_dataset = WN18Dataset(data_path=self.data_path, dataset_name='test', entity2id=self.entity2id, relation2id=self.relation2id)
        # init entity / relation embeddings
        self.entity_embeddings = nn.Embedding(self.num_entity, args.embedding_dim)
        self.relation_embeddings = nn.Embedding(self.num_relation, args.embedding_dim)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=args.factor)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        h, r, t, v = batch
        h = self.entity_embeddings(h)
        r = self.relation_embeddings(r)
        t = self.entity_embeddings(t)
        conv_input = torch.cat(
            (
                h.unsqueeze(1),
                r.unsqueeze(1),
                t.unsqueeze(1),
            ),
            dim=1,
        )
        conv_output = self(conv_input.to(device=self.device))
        conv_loss = calc_conv_loss(self.model, batch.to(device=self.device), conv_output, len(e1))
        self.log('train_loss', conv_loss, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': conv_loss}

    def validation_step(self, batch, batch_idx):
        pass

    def test_step(self, batch, batch_idx):
        pass


if __name__ == '__main__':
    model1 = PreTrainGCN()
    model2 = TrainConvKB()
    trainer = Trainer(
        gpus=1,
        accelerator='dp',
        max_epochs=args.epoch_size,
        logger=TensorBoardLogger(
            save_dir='.',
            name='log',
        )
    )
    trainer.fit(model1)
