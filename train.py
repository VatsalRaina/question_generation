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
from transformers import AdamW, ElectraConfig
from transformers import get_linear_schedule_with_warmup

MAXLEN_passage = 768
MAXLEN_question = 768

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=2, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=10, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')

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

    # Set the seed value all over the place to make this reproducible.
    seed_val = args.seed
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)
    # Choose device
    device = get_default_device()

    train_data = load_dataset('squad_v2', split='train')

    # Load the GPT tokenizer.
    tokenizer = GPT2Tokenizer.from_pretrained('gpt2', bos_token='<|startoftext|>', eos_token='<|endoftext|>', pad_token='<|pad|>') #gpt2-medium
    # The max model length is 1024 for this model, although the actual embedding size for GPT small is 768
    # The beginning of sequence token <|startoftext|> token has the id 50258
    # The end of sequence token <|endoftext|> has the id 50256
    # The padding token <|pad|> has the id 50257

    input_ids = []
    output_ids = []
    input_att_msks = []
    count = 0

    for ex in train_data:
        if len(ex["answers"]["text"])==0:
            continue
        count+=1
        # if count==100:
        #     break
        print(count)
        question, passage = ex["question"], ex["context"]
        passage_encodings_dict = tokenizer('<|startoftext|>'+ passage + '<|endoftext|>', truncation=True, max_length=MAXLEN_passage, padding="max_length")
        input_ids.append(passage_encodings_dict['input_ids'])
        input_att_msks.append(passage_encodings_dict['attention_mask'])
        question_encodings_dict = tokenizer('<|startoftext|>'+ question + '<|endoftext|>', truncation=True, max_length=MAXLEN_question, padding="max_length")
        output_ids.append(question_encodings_dict['input_ids'])

    # Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    input_att_msks = torch.tensor(input_att_msks)
    input_att_msks = input_att_msks.long().to(device)
    output_ids = torch.tensor(output_ids)
    output_ids = output_ids.long().to(device)
    
    train_data = TensorDataset(input_ids, input_att_msks, output_ids)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # I'm not really doing anything with the config buheret
    configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)

    # instantiate the model
    model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)

    # this step is necessary because I've added some tokens (bos_token, etc) to the embeddings
    # otherwise the tokenizer and model tensors won't match up
    model.resize_token_embeddings(len(tokenizer))

    model.to(device)

    optimizer = AdamW(model.parameters(),
                    lr = args.learning_rate,
                    eps = args.adam_epsilon
                    # weight_decay = 0.01
                    )

    loss_values = []

    # Total number of training steps is number of batches * number of epochs.
    total_steps = len(train_dataloader) * args.n_epochs
    # Create the learning rate scheduler.
    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps = 0.1*total_steps,
                                                num_training_steps = total_steps)

    for epoch in range(args.n_epochs):
        # Perform one full pass over the training set.
        print("")
        print('======== Epoch {:} / {:} ========'.format(epoch + 1, args.n_epochs))
        print('Training...')
        # Measure how long the training epoch takes.
        t0 = time.time()
        # Reset the total loss for this epoch.
        total_loss = 0
        model.train()
    # For each batch of training data...
        for step, batch in enumerate(train_dataloader):
            # Progress update every 40 batches.
            if step % 40 == 0 and not step == 0:
                # Calculate elapsed time in minutes.
                elapsed = format_time(time.time() - t0)
                # Report progress.
                print('  Batch {:>5,}  of  {:>5,}.    Elapsed: {:}.'.format(step, len(train_dataloader), elapsed))
            b_input_ids = batch[0].to(device)
            b_att_msks = batch[1].to(device)
            b_output_ids = batch[2].to(device)
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, labels=b_output_ids, token_type_ids=None)
            loss = outputs[0]
            print(loss.item())
            total_loss += loss.item()
            loss.backward()
            # Clip the norm of the gradients to 1.0.
            # This is to help prevent the "exploding gradients" problem.
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            # Update parameters and take a step using the computed gradient.
            # The optimizer dictates the "update rule"--how the parameters are
            # modified based on their gradients, the learning rate, etc.
            optimizer.step()
            # Update the learning rate.
            scheduler.step()
        # Calculate the average loss over the training data.
        avg_train_loss = total_loss / len(train_dataloader)

        # Store the loss value for plotting the learning curve.
        loss_values.append(avg_train_loss)

        print("")
        print("  Average training loss: {0:.2f}".format(avg_train_loss))
        print("  Training epoch took: {:}".format(format_time(time.time() - t0)))

    # Save the model to a file
    file_path = args.save_path+'gpt2_gen_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)


if __name__ == '__main__':
    args = parser.parse_args()
    main(args)