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
from transformers import GPT2LMHeadModel,  GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW
from transformers import get_linear_schedule_with_warmup

MAXLEN_passage = 400
MAXLEN_question = 400

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=32, help='Specify the training batch size')
parser.add_argument('--model_path', type=str, help='Load path of trained model')
parser.add_argument('--predictions_save_path', type=str, help='Load path to which predicted values')

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

    # Load the GPT tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2-large', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-large
    # The max model length is 1024 for this model, although the actual embedding size for GPT small is 768
    # The beginning of sequence token <|startoftext|> token has the id 50258
    # The end of sequence token <|endoftext|> has the id 50256
    # The padding token <|pad|> has the id 50257

    input_ids = []
    output_ids = []
    input_att_msks = []
    count = 0

    model = torch.load(args.model_path, map_location=device)
    model.eval().to(device)

    for ex in test_data:
        if len(ex["answers"]["text"])==0:
            continue
        count+=1
        if count==2:
            break
        print(count)
        print(" ")
        question, passage = ex["question"], ex["context"]
        print(question)
        print(passage)
        print("-------")

        generated = torch.tensor(tokenizer.encode('<|startoftext|>'+ passage + '<|endoftext|>')).unsqueeze(0)
        generated = generated.to(device)

        sample_outputs = model.generate(
                                generated, 
                                #bos_token_id=random.randint(1,30000),
                                do_sample=True,   
                                top_k=50, 
                                max_length = 300,
                                top_p=0.95, 
                                num_return_sequences=3
                                )

        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}\n\n".format(i, tokenizer.decode(sample_output, skip_special_tokens=True)))

        # passage_encodings_dict = tokenizer('<|startoftext|>'+ passage + '<|endoftext|>', truncation=True, max_length=MAXLEN_passage, padding="max_length")
        # input_ids.append(passage_encodings_dict['input_ids'])
        # input_att_msks.append(passage_encodings_dict['attention_mask'])
        # question_encodings_dict = tokenizer('<|startoftext|>'+ question + '<|endoftext|>', truncation=True, max_length=MAXLEN_question, padding="max_length")
        # output_ids.append(question_encodings_dict['input_ids'])

    # # Convert to torch tensors
    # input_ids = torch.tensor(input_ids)
    # input_ids = input_ids.long().to(device)
    # input_att_msks = torch.tensor(input_att_msks)
    # input_att_msks = input_att_msks.long().to(device)
    # output_ids = torch.tensor(output_ids)
    # output_ids = output_ids.long().to(device)
    
    # test_data = TensorDataset(input_ids, input_att_msks, output_ids)
    # test_dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False)

    # model = torch.load(args.model_path, map_location=device)
    # model.eval().to(device)

    # predictions = []


    # for step, batch in enumerate(test_dataloader):
    #     b_input_ids = batch[0].to(device)
    #     b_att_msks = batch[1].to(device)
    #     b_output_ids = batch[2].to(device)
    #     with torch.no_grad():
    #         outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, labels=b_output_ids, token_type_ids=None)
    #     loss = outputs[0]
    #     print(loss.item())




if __name__ == '__main__':
    args = parser.parse_args()
    main(args)