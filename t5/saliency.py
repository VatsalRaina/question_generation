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
    passage = "Because of this, Ethelred fled to Normandy in 1013, when he was forced from his kingdom by Sweyn Forkbeard."
    
    passage_encodings_dict = tokenizer(passage, return_tensors="pt")
    inp_id = passage_encodings_dict['input_ids']

    question_encodings_dict = tokenizer(gen_question, return_tensors="pt")
    out_id = question_encodings_dict['input_ids']

    embedding_matrix = model.shared
    embedded = torch.tensor(embedding_matrix(inp_id), requires_grad=True)

    outputs = model(inputs_embeds=embedded, labels=out_id)
    loss = outputs[0]

    loss.backward()

    saliency_max = torch.squeeze(torch.norm(embedded.grad.data.abs(), dim=-1))
    saliency_max = saliency_max.detach().cpu().numpy()
    # remove end of sentence token
    saliency_max = saliency_max[:-1]

    words = tokenizer.tokenize(passage)

    M = len(words)
    print(M)
    print(len(list(saliency_max)))
    xx = np.linspace(0, M, M)
    plt.figure(figsize=(40,40))
    plt.barh(xx, list(saliency_max)[::-1])
    plt.yticks(xx, labels=np.flip([w[1:] for w in words if w[0]=='_' else w]), fontsize=40)
    plt.xticks(fontsize=40)
    plt.ylabel('Passage')
    plt.ylim([-2, M+2])
    #plt.xlim([0.0, 0.17])
    plt.savefig('./saliencyQu.png')
    plt.close()
 
if __name__ == '__main__':
    args = parser.parse_args()
    main(args)