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
parser.add_argument('--num_beams', type=int, default=1, help='Number of beams to use')

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

    test_data = load_dataset('squad_v2', split='validation')

    tokenizer = T5Tokenizer.from_pretrained("t5-base")

    count = 0

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    prev_passage = ""

    all_passages = []
    all_generated_questions = []

    for ex in test_data:
        if len(ex["answers"]["text"])==0:
            continue
        question, passage = ex["question"], ex["context"].replace('\n', '')
        if passage==prev_passage:
            continue
        prev_passage=passage
        count+=1
        # if count==20:
        #     break
        
        #print(" ")
        print(count)
        # print(question)
        #print(passage)
        #print("Here is the generated question:")
        passage_encodings_dict = tokenizer(passage, truncation=True, max_length=MAXLEN_passage, padding="max_length", return_tensors="pt")
        inp_id = passage_encodings_dict['input_ids']
        inp_att_msk = passage_encodings_dict['attention_mask']

        all_generated_ids = model.generate(
            input_ids=inp_id,
            attention_mask=inp_att_msk,
            num_beams=args.num_beams, 
            do_sample=False,
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
            # preds = [
            #     tokenizer.decode(generated_id, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            #     for generated_id in generated_ids
            # ]
            # all_generated_questions.append("".join(preds))

            genQu = tokenizer.decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)
            all_generated_questions.append(genQu)
            all_passages.append(passage)

    #print(len(all_passages))

    with open("passages.txt", 'w') as f:
        f.writelines("%s\n" % passag for passag in all_passages)

    with open("gen_questions.txt", 'w') as f:
        f.writelines("%s\n" % qu for qu in all_generated_questions)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)