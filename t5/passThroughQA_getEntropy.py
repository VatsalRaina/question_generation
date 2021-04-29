#! /usr/bin/env python

import argparse
import os
import sys

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
from sklearn.metrics import precision_recall_curve
from scipy.stats import entropy
import random
import time
import datetime

from datasets import load_dataset
from transformers import ElectraTokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

from models import ElectraQA

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--model_path', type=str, help='Load path to trained model')
parser.add_argument('--predictions_save_path', type=str, help="Where to save predicted values")
parser.add_argument('--questions_path', type=str, help='Path to questions')
parser.add_argument('--passages_path', type=str, help='Path to passages')


def format_time(elapsed):
    '''
    Takes a time in seconds and returns a string hh:mm:ss
    '''
    # Round to the nearest second.
    elapsed_rounded = int(round((elapsed)))

    # Format as hh:mm:ss
    return str(datetime.timedelta(seconds=elapsed_rounded))

# Set device
def get_default_device():
    # Force cpu
    return torch.device('cpu')
    if torch.cuda.is_available():
        print("Got CUDA!")
        return torch.device('cuda')
    else:
        return torch.device('cpu')

def main(args):
    if not os.path.isdir('CMDs'):
        os.mkdir('CMDs')
    with open('CMDs/train.cmd', 'a') as f:
        f.write(' '.join(sys.argv) + '\n')
        f.write('--------------------------------\n')

    # Choose device
    device = get_default_device()

    with open(args.passages_path, 'r') as f:
        all_passages = [a.rstrip() for a in f.readlines()]

    with open(args.questions_path, 'r') as f:
        all_questions = [a.rstrip() for a in f.readlines()]


    electra_base = "google/electra-base-discriminator"
    electra_large = "google/electra-large-discriminator"
    tokenizer = ElectraTokenizer.from_pretrained(electra_large, do_lower_case=True)


    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    entropy_all = []

    print(len(all_passages))
    for i in range(len(all_passages)):
        # if i==20:
        #     break
        print(i)
        passage = all_passages[i]
        question = all_questions[i]
        combo = question + " [SEP] " + passage
        inp_ids = tokenizer.encode(combo)
        if len(inp_ids) > 512:
            inp_ids = inp_ids[:512]
        pr_resp_pt = torch.tensor(inp_ids).to(device)
        embedding_matrix = model.electra.embeddings.word_embeddings
        embedded = torch.tensor(embedding_matrix(pr_resp_pt), requires_grad=True)

        start_logits, end_logits, _ = model.saliency(torch.unsqueeze(embedded, 0))
        sum_logits = torch.sum(start_logits + end_logits)
        sum_logits.backward()

        saliency_av = torch.norm(embedded.grad.data.abs(), dim=1)
        saliency_av = saliency_av.detach().cpu().numpy()

        # We don't care about the first and last tokens
        saliency_av = saliency_av[1:-1]

        # Extract only the response words
        words = tokenizer.tokenize(combo)
        sep = words.index("[SEP]")
        saliency_av = saliency_av[sep+1:]

        # Normalise values
        saliency_av = saliency_av / np.sum(saliency_av)

        entrop = entropy(saliency_av, base=2)
        entropy_all.append(entrop)

    entropy_all = np.asarray(entropy_all)
    np.save(args.predictions_save_path + "sal_entrop.npy", entropy_all)
        

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)
