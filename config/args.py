import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-info', '--info', type=str, default='', help='comment on this training.')
    args.add_argument('-mode', '--mode', type=str, choices=['train', 'test'], default='train')
    args.add_argument('-dataset', '--dataset', type=str, choices=['FB15k-237', 'kinship', 'NELL-995', 'umls', 'WN18RR'], default='umls')
    args.add_argument('-pre_train_model', '--pre_train_model', type=str, choices=['ours', 'baseline', 'none'], default='none')
    args.add_argument('-epoch_size', '--epoch_size', type=int, default=64, help='epochs number')
    args.add_argument('-embedding_dim', '--embedding_dim', type=int, default=32, help='embedding dimension')
    args.add_argument('-train_batch_size', '--train_batch_size', type=int, default=256, help='train batch size.')
    args.add_argument('-val_batch_size', '--val_batch_size', type=int, default=4096, help='validation batch size.')
    args.add_argument('-test_batch_size', '--test_batch_size', type=int, default=4096, help='test batch size.')
    args.add_argument('-lr', '--lr', type=float, default=0.01, help='initial learning rate.')
    args.add_argument('-factor', '--factor', type=float, default=0.9, help='reduce learning rate factor.')
    args.add_argument('-num_workers', '--num_workers', type=int, default=8, help='dataloader workers number')
    args.add_argument('-dropout', '--dropout', type=float, default=0.2, help='dropout probability')
    args.add_argument('-margin', '--margin', type=float, default=0.5, help='for margin ranking loss')
    args.add_argument('-hits', '--hits', type=float, default=10, help='for evaluation')
    args.add_argument('-step_size', '--step_size', type=float, default=50, help='for lr scheduler')
    args.add_argument('-out_channels', '--out_channels', type=int, default=32, help='for ConvE')
    return args.parse_args()


args = parse_args()
