import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F

import pytorch_lightning as pl
from pytorch_lightning import seed_everything, Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from config.args import args
from data.dataset import KGDataset
from model.model import RandomResidualGCN, DoubleGCN, ConvE
from util.util import read_id, generate_adj, get_head_relation_tail, align, get_checkpoint


class TrainKGE(pl.LightningModule):
    def __init__(self, init):
        super().__init__()
        self.data_path = init['data_path']
        self.entity2id = init['entity2id']
        self.relation2id = init['relation2id']
        self.num_entity = init['num_entity']
        self.num_relation = init['num_relation']
        self.init_emb_e = init['init_entity_embeddings'].to(self.device)
        self.init_emb_rel = init['init_relation_embeddings'].to(self.device)
        # TODO init model

    def prepare_data(self):
        self.train_dataset = KGDataset(data_path=self.data_path, dataset_name='train', entity2id=self.entity2id, relation2id=self.relation2id, has_negative_data=False)
        self.val_dataset = KGDataset(data_path=self.data_path, dataset_name='valid', entity2id=self.entity2id, relation2id=self.relation2id, has_negative_data=False)
        self.test_dataset = KGDataset(data_path=self.data_path, dataset_name='test', entity2id=self.entity2id, relation2id=self.relation2id, has_negative_data=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=args.train_batch_size, num_workers=args.num_workers, pin_memory=True, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=args.val_batch_size, num_workers=args.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=args.test_batch_size, num_workers=args.num_workers, pin_memory=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=args.lr)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, args.factor)
        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):
        raise NotImplementedError

    def validation_step(self, batch, batch_idx):
        h, r, t, v = batch
        acc, mr, mrr = self.model.evaluate(h, r, t)
        self.log('acc', acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('mr', mr, on_epoch=True, prog_bar=True, logger=True)
        self.log('mrr', mrr, on_epoch=True, prog_bar=True, logger=True)
        return {'acc': acc, 'mr': mr, 'mrr': mrr}

    def test_step(self, batch, batch_idx):
        h, r, t, v = batch
        acc, mr, mrr = self.model.evaluate(h, r, t)
        self.log('acc', acc, on_epoch=True, prog_bar=True, logger=True)
        self.log('mr', mr, on_epoch=True, prog_bar=True, logger=True)
        self.log('mrr', mrr, on_epoch=True, prog_bar=True, logger=True)
        return {'acc': acc, 'mr': mr, 'mrr': mrr}

    @property
    def entity_embeddings(self):
        raise NotImplementedError

    @property
    def relation_embeddings(self):
        raise NotImplementedError


class PreTrainGCN(TrainKGE):
    def __init__(self, init):
        super().__init__(init)
        # TODO init GCN model

    def forward(self, h, r, t, v, adj, loss_function):
        return self.model(h, r, t, v, adj, loss_function)

    def prepare_data(self):
        super().prepare_data()
        # re-init train data (needs negative pairs if use margin ranking loss, default: False)
        # self.train_dataset = KGDataset(data_path=self.data_path, dataset_name='train', entity2id=self.entity2id, relation2id=self.relation2id, has_negative_data=True)
        # init train / validation / test adjacency matrix
        self.train_adj = generate_adj(data=self.train_dataset.data, num_entity=self.num_entity)
        self.val_adj = generate_adj(data=self.val_dataset.data, num_entity=self.num_entity)
        self.test_adj = generate_adj(data=self.test_dataset.data, num_entity=self.num_entity)

    def training_step(self, batch, batch_idx):
        h, r, t, v = batch
        loss = self(h, r, t, v, self.train_adj, self.calc_mse_loss)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=True, logger=True)
        return {'loss': loss}

    def calc_margin_loss(self, triples, values):
        pos_triples, pos_len = get_head_relation_tail(triples, values, 'positive')
        neg_triples, neg_len = get_head_relation_tail(triples, values, 'negative')

        # alignment
        if pos_len > neg_len:
            neg_triples = align(neg_triples, neg_len, pos_len)
        elif neg_len > pos_len:
            pos_triples = align(pos_triples, pos_len, neg_len)

        pos_h, pos_r, pos_t = pos_triples
        neg_h, neg_r, neg_t = neg_triples

        # norm
        pos_norm = torch.norm(pos_h + pos_r - pos_t, p=1, dim=1)
        neg_norm = torch.norm(neg_h + neg_r - neg_t, p=1, dim=1)

        # margin ranking loss
        return F.margin_ranking_loss(
            pos_norm,
            neg_norm,
            torch.ones(len(pos_norm), device=self.device, requires_grad=True),
            margin=args.margin
        )

    def calc_mse_loss(self, triples, values):
        h, r, t = triples
        return F.mse_loss(h + r - t, torch.zeros((h.size(0), args.embedding_dim), device=h.device).float())

    @property
    def entity_embeddings(self):
        return self.model.entity_embeddings

    @property
    def relation_embeddings(self):
        return self.model.relation_embeddings


class PreTrainRandomResidualGCN(PreTrainGCN):
    def __init__(self, init):
        super().__init__(init)
        self.model = RandomResidualGCN(self.num_entity, self.num_relation, self.init_emb_e, self.init_emb_rel)


class PreTrainDoubleGCN(PreTrainGCN):
    def __init__(self, init):
        super().__init__(init)
        self.model = DoubleGCN(self.num_entity, self.num_relation, self.init_emb_e, self.init_emb_rel)


class TrainConvE(TrainKGE):
    def __init__(self, init):
        super().__init__(init)
        self.model = ConvE(self.num_entity, self.num_relation, init_emb_e=self.init_emb_e, init_emb_rel=self.init_emb_rel)

    def forward(self, h, r, t):
        return self.model(h, r, t)

    def training_step(self, batch, batch_idx):
        h, r, t, v = batch
        _, loss = self(h, r, t)
        self.log('train_loss', loss, on_step=False, on_epoch=True, prog_bar=False, logger=True)
        return {'loss': loss}

    @property
    def entity_embeddings(self):
        return self.model.emb_e

    @property
    def relation_embeddings(self):
        return self.model.emb_rel


def get_trainer():
    return Trainer(
        gpus=1,
        auto_select_gpus=True,
        accelerator='dp',
        max_epochs=args.epoch_size,
        logger=TensorBoardLogger(
            save_dir='.',
            name='log',
        ),
        # callbacks=[EarlyStopping(monitor='mr', min_delta=0.0, patience=3, verbose=True)],
        checkpoint_callback=ModelCheckpoint(save_top_k=1, monitor='mr'),
        check_val_every_n_epoch=1,
    )


def get_init(data_path, entity2id, relation2id, num_entity, num_relation, init_entity_embeddings, init_relation_embeddings):
    return {
        'data_path': data_path,
        'entity2id': entity2id,
        'relation2id': relation2id,
        'num_entity': num_entity,
        'num_relation': num_relation,
        'init_entity_embeddings': init_entity_embeddings,
        'init_relation_embeddings': init_relation_embeddings,
    }


if __name__ == '__main__':
    seed_everything(42)

    # init
    data_path = os.path.join('data', args.dataset)
    entity2id = read_id(os.path.join(data_path, 'entity2id.txt'))
    relation2id = read_id(os.path.join(data_path, 'relation2id.txt'))
    num_entity = len(entity2id)
    num_relation = len(relation2id)
    init_entity_embeddings = nn.Embedding(num_entity, args.embedding_dim)
    init_relation_embeddings = nn.Embedding(num_relation, args.embedding_dim)

    # pre train model (if not none)
    if args.pre_train_model != 'none':
        init = get_init(data_path, entity2id, relation2id, num_entity, num_relation, init_entity_embeddings, init_relation_embeddings)
        if args.pre_train_model == 'ours':
            pre_train_model = PreTrainRandomResidualGCN(init)
        if args.pre_train_model == 'baseline':
            pre_train_model = PreTrainDoubleGCN(init)
        trainer = get_trainer()
        trainer.fit(pre_train_model)
        trainer.test(pre_train_model)
        init_entity_embeddings = pre_train_model.entity_embeddings
        init_relation_embeddings = pre_train_model.relation_embeddings

    # train main model
    init = get_init(data_path, entity2id, relation2id, num_entity, num_relation, init_entity_embeddings, init_relation_embeddings)
    main_model = TrainConvE(init)
    trainer = get_trainer()
    trainer.fit(main_model)
    trainer.test(main_model)
