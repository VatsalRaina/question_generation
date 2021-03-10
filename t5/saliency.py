import argparse
import os
import sys
import json

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
import numpy as np
import random
import time
import datetime

from datasets import load_dataset
from transformers import T5ForConditionalGeneration, T5Tokenizer
from keras.preprocessing.sequence import pad_sequences


MAXLEN_passage = 400
MAXLEN_question = 100

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=2, help='Specify the training batch size')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--prediction_save_path', type=str, help='Load path to which trained model will be saved')

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
    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)
    model.zero_grad()


    gen_question = 
    passage = 
    passage_encodings_dict = tokenizer(passage, return_tensors="pt")
    inp_id torch.unsqueeze(inp_id = passage_encodings_dict['input_ids'], 0)

    question_encodings_dict = tokenizer(question, return_tensors="pt")
    out_id = torch.unsqueeze(passage_encodings_dict['input_ids'], 0)

    embedding_matrix = model.t5.embeddings.word_embeddings
    embedded = torch.tensor(embedding_matrix(inp_id), requires_grad=True)

    outputs = model(input_ids=inp_id, labels=out_id)
    loss = outputs[0]

    loss.backward()

    saliency_max = torch.squeeze(torch.norm(embedded.grad.data.abs(), dim=-1))
    saliency_max = saliency_max.detach().cpu().numpy()

    words = tokenizer.tokenize(passage)

    M = len(words)
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(40,80))
    plt.barh(xx, list(saliency_max)[::-1])
    plt.yticks(xx, labels=np.flip(words), fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Passage')
    plt.ylim([-2, M+2])
    #plt.xlim([0.0, 0.17])
    plt.savefig('./saliencyQu.png')
    plt.close()
 
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)