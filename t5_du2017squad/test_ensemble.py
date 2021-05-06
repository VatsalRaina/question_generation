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

from generate_ensemble import EnsembleModel

MAXLEN_sentence = 100
MAXLEN_question = 100

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=2, help='Specify the training batch size')
parser.add_argument('--model1_path', type=str, help='Load path of trained model 1')
parser.add_argument('--model2_path', type=str, help='Load path of trained model 2')
parser.add_argument('--prediction_save_path', type=str, help='Load path to which trained model will be saved')
parser.add_argument('--num_beams', type=int, default=1, help='Number of beams')
parser.add_argument('--sentence_path', type=str, help='Load path to testing sentences')
parser.add_argument('--question_path', type=str, help='Load path to testing questions')


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

    sentence_file = open(args.sentence_path, "r")
    sentences = sentence_file.readlines()
    question_file = open(args.question_path, "r")
    questions = question_file.readlines()    

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    count = 0

    model1 = torch.load(args.model1_path, map_location=device)
    model1.eval().to(device)

    model2 = torch.load(args.model2_path, map_location=device)
    model2.eval().to(device)

    ensembleModel = EnsembleModel([model1, model2])

    all_sentences = []
    all_generated_questions = []

    for sentence, question in zip(sentences, questions):
        sentence = sentence.replace('\n', '')
        if sentence in all_sentences:
            continue
        count+=1
        # if count==20:
        #     break
        print(count)
        sentence_encodings_dict = tokenizer(sentence, truncation=True, max_length=MAXLEN_sentence, padding="max_length", return_tensors="pt")
        inp_id = sentence_encodings_dict['input_ids']
        inp_att_msk = sentence_encodings_dict['attention_mask']

        all_generated_ids = ensembleModel.generate(
            input_ids=inp_id,
            attention_mask=inp_att_msk,
            num_beams=args.num_beams,
            #do_sample=True,
            #top_k=50,           # This parameter and the one below create more question variability but reduced quality of questions
            #top_p=0.95,          
            max_length=80,
            repetition_penalty=2.5,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            num_return_sequences=1
        )
        #print(len(all_generated_ids))
        for generated_ids in all_generated_ids:
            genQu = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            all_generated_questions.append(genQu)
            all_sentences.append(sentence)

    #print(len(all_passages))

    with open("sentences.txt", 'w') as f:
        f.writelines("%s\n" % sentec for sentec in all_sentences)

    with open("gen_questions.txt", 'w') as f:
        f.writelines("%s\n" % qu for qu in all_generated_questions)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)