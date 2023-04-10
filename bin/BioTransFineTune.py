import random
import pickle
from biotransformers import BioTransformers
import torch
import ray
from sklearn.model_selection import train_test_split
import argparse


def parse_args():
    parser = argparse.ArgumentParser(description='Biotransformer Finetuning')
    parser.add_argument('--log_name', type=str, help='Name for folder of finetuned biotrans model')
    parser.add_argument('--train_path', type=str, help='path to pickled training set')
    parser.add_argument('--val_split', type=float, default=0.2,
                        help='percentage of sequences for validation')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs')
    parser.add_argument('--model_type', type=str, help='Type of biotransformer model', default='protbert')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = parse_args()

    random.seed(42)

    with open(args.train_path, 'rb') as f:
        train = pickle.load(f)

    if isinstance(train, dict):
        seqs = train['train']
    else:
        seqs = train

    print("N SARS-CoV-2 Seqs: {}".format(len(seqs)))

    random.shuffle(seqs)

    print("N seqs: ", str(len(seqs)))
    seqs = list(set(seqs))
    print("N Unique seqs: ", str(len(seqs)))

    split_perc = args.val_split
    print("Validation split percentage: {}".format(split_perc))

    train, val = train_test_split(seqs, random_state=42, test_size=split_perc)

    print("N training: ", str(len(train)))
    print("N validation : ", str(len(val)))


    print('Loading biotransformer')

    n_gpu = torch.cuda.device_count()
    print("N GPU: {}".format(n_gpu))
    ray.init()
    bio_trans = BioTransformers(backend=args.model_type, num_gpus=n_gpu)

    print()
    print(bio_trans.__dict__)
    print()

   #chkpoint = 'logs/protbert_cv_ft_v1/version_3/checkpoints/epoch=14-step=174.ckpt'
    chkpoint = None
    print('fine tuning')
    bio_trans.finetune(train, validation_sequences=val, accelerator='ddp', logs_save_dir='logs',
                       checkpoint=chkpoint, save_last_checkpoint=False, amp_level=None,
                       logs_name_exp=args.log_name,  epochs=args.epochs)


    print('Done ')
    