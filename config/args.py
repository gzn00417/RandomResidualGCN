import argparse


def parse_args():
    args = argparse.ArgumentParser()
    args.add_argument('-info', '--info', type=str, default='', help='comment on this training.')
    args.add_argument('-mode', '--mode', type=str, choices=['train', 'test'], default='train')
    args.add_argument('-epoch_size', '--epoch_size', type=int, default=256, help='epochs number')
    args.add_argument('-embedding_dim', '--embedding_dim', type=int, default=256, help='embedding dimension')
    args.add_argument('-train_batch_size', '--train_batch_size', type=int, default=4096, help='train batch size.')
    args.add_argument('-val_batch_size', '--val_batch_size', type=int, default=4096, help='validation batch size.')
    args.add_argument('-test_batch_size', '--test_batch_size', type=int, default=1024, help='test batch size.')
    args.add_argument('-lr', '--lr', type=float, default=0.1, help='initial learning rate.')
    args.add_argument('-factor', '--factor', type=float, default=0.8, help='reduce learning rate factor.')
    args.add_argument('-num_workers', '--num_workers', type=int, default=1, help='dataloader workers number')
    args.add_argument('-dropout', '--dropout', type=float, default=0.5, help='dropout probability')
    args.add_argument('-margin', '--margin', type=float, default=0.5, help='for margin ranking loss')
    args.add_argument('-lmbda', '--lmbda', type=float, default=0.1, help='lambda for convkb loss')
    args.add_argument('-hits_scale', '--hits_scale', type=float, default=0.1, help='')
    return args.parse_args()


args = parse_args()
