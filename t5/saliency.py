import argparse
import os
import sys

import torch
import numpy as np
from transformers import T5ForConditionalGeneration, T5Tokenizer

import matplotlib
matplotlib.use('agg')
from matplotlib import pyplot as plt

MAXLEN_passage = 400
MAXLEN_question = 100

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--model_path', type=str, help='Load path of trained model')

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


    gen_question = "Who forced Ethelred II from his kingdom?"
    passage = "The Normans were in contact with England from an early date. Not only were their original Viking brethren still ravaging the English coasts, they occupied most of the important ports opposite England across the English Channel. This relationship eventually produced closer ties of blood through the marriage of Emma, sister of Duke Richard II of Normandy, and King Ethelred II of England. Because of this, Ethelred fled to Normandy in 1013, when he was forced from his kingdom by Sweyn Forkbeard. His stay in Normandy (until 1016) influenced him and his sons by Emma, who stayed in Normandy after Cnut the Great’s conquest of the isle."
    
    passage_encodings_dict = tokenizer(passage, return_tensors="pt")
    inp_id = torch.unsqueeze(passage_encodings_dict['input_ids'], 0)

    question_encodings_dict = tokenizer(gen_question, return_tensors="pt")
    out_id = torch.unsqueeze(question_encodings_dict['input_ids'], 0)

    embedding_matrix = model.t5.embeddings.word_embeddings
    embedded = torch.tensor(embedding_matrix(inp_id), requires_grad=True)

    outputs = model(input_ids=inp_id, labels=out_id)
    loss = outputs[0]

    loss.backward()

    saliency_max = torch.squeeze(torch.norm(embedded.grad.data.abs(), dim=-1))
    saliency_max = saliency_max.detach().cpu().numpy()

    words = tokenizer.tokenize(passage)

    M = len(words)
    print(M)
    print(len(list(saliency_max)))
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