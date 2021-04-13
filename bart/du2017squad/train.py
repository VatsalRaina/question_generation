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

from transformers import BartForConditionalGeneration, BartTokenizer
from keras.preprocessing.sequence import pad_sequences
from transformers import AdamW, BartConfig
from transformers import get_linear_schedule_with_warmup

MAXLEN_sentence = 100
MAXLEN_question = 100

parser = argparse.ArgumentParser(description='Get all command line arguments.')
parser.add_argument('--batch_size', type=int, default=2, help='Specify the training batch size')
parser.add_argument('--learning_rate', type=float, default=2e-5, help='Specify the initial learning rate')
parser.add_argument('--adam_epsilon', type=float, default=1e-6, help='Specify the AdamW loss epsilon')
parser.add_argument('--lr_decay', type=float, default=0.85, help='Specify the learning rate decay rate')
parser.add_argument('--dropout', type=float, default=0.1, help='Specify the dropout rate')
parser.add_argument('--n_epochs', type=int, default=10, help='Specify the number of epochs to train for')
parser.add_argument('--seed', type=int, default=1, help='Specify the global random seed')
parser.add_argument('--save_path', type=str, help='Load path to which trained model will be saved')
parser.add_argument('--sentence_path', type=str, help='Load path to training sentences')
parser.add_argument('--question_path', type=str, help='Load path to training questions')


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

    sentence_file = open(args.sentence_path, "r")
    sentences = sentence_file.readlines()
    question_file = open(args.question_path, "r")
    questions = question_file.readlines()    

    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large", add_prefix_space=True)

    input_ids = []
    output_ids = []
    input_att_msks = []
    output_att_msks = []
    count = 0

    for sentence, question in zip(sentences, questions):
        count+=1
        # if count==10:
        #     break
        print(count)
        sentence_encodings_dict = tokenizer(sentence, truncation=True, max_length=MAXLEN_sentence, padding="max_length")
        input_ids.append(sentence_encodings_dict['input_ids'])
        input_att_msks.append(sentence_encodings_dict['attention_mask'])
        question_encodings_dict = tokenizer(question, truncation=True, max_length=MAXLEN_question, padding="max_length")
        output_ids.append(question_encodings_dict['input_ids'])
        output_att_msks.append(question_encodings_dict['attention_mask'])        
        # output_ids.append([x if x!=0 else -100 for x in question_encodings_dict['input_ids']])

    # Convert to torch tensors
    input_ids = torch.tensor(input_ids)
    input_ids = input_ids.long().to(device)
    input_att_msks = torch.tensor(input_att_msks)
    input_att_msks = input_att_msks.long().to(device)
    output_ids = torch.tensor(output_ids)
    output_ids = output_ids.long().to(device)
    output_att_msks = torch.tensor(output_att_msks)
    output_att_msks = output_att_msks.long().to(device)

    train_data = TensorDataset(input_ids, input_att_msks, output_ids, output_att_msks)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=args.batch_size)

    # instantiate the model
    model = BartForConditionalGeneration.from_pretrained("facebook/bart-large")


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
            b_target_ids = batch[2].to(device)
            b_out_att_msks = batch[3].to(device)
            b_output_ids = b_target_ids[:, :-1].contiguous()
            b_labels = b_target_ids[:, 1:].clone()
            b_labels[b_target_ids[:, 1:] == 0] = -100
            model.zero_grad()
            outputs = model(input_ids=b_input_ids, attention_mask=b_att_msks, decoder_input_ids=b_output_ids, decoder_attention_mask=b_out_att_msks, labels=b_labels)
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
    file_path = args.save_path+'bart_gen_seed'+str(args.seed)+'.pt'
    torch.save(model, file_path)

if __name__ == '__main__':
    args = parser.parse_args()
    main(args)