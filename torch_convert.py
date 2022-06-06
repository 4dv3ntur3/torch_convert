import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

from utils import init_logger, load_tokenizer


def get_args(pred_config):
    return torch.load(os.path.join(pred_config, 'training_params.bin'))


def read_input_file(input_dir):
    lines = []
    with open(input_dir, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         args,
                                         tokenizer,
                                         pad_token_label_id,
                                         cls_token_segment_id=0,
                                         pad_token_segment_id=0,
                                         sequence_a_segment_id=0,
                                         mask_padding_with_zero=True):
    # Setting based on the current model type
    cls_token = tokenizer.cls_token
    sep_token = tokenizer.sep_token
    unk_token = tokenizer.unk_token
    pad_token_id = tokenizer.pad_token_id

    all_input_ids = []
    all_attention_mask = []
    all_token_type_ids = []
    all_slot_label_mask = []

    all_input_tokens = []
    for words in lines:
        tokens = []
        slot_label_mask = []
        for word in words:
            word_tokens = tokenizer.tokenize(word)
            if not word_tokens:
                word_tokens = [unk_token]  # For handling the bad-encoded word
            tokens.extend(word_tokens)
            
            # Use the real label id for the first token of the word, and padding ids for the remaining tokens
            # slot_label_mask.extend([0] + [pad_token_label_id] * (len(word_tokens) - 1))
            
            # use the real label id for all tokens of the word
            slot_label_mask.extend([0] * (len(word_tokens)))

        # print(tokens)
        all_input_tokens.append(tokens) # 뭐지? 여기서는 뒤에 sep 안 붙는데 이 이후부터 계속 붙네?
        # append를 여기서 하는데 왜지??? 
        # print("=====================")
        # print(all_input_tokens)
        # 어차피 input_tokens 뒤에 SEP가 붙어서 +1이더라도 preds_list는 그거보다 1개 작으니까 (SEP 없음) 상관은 없는데
        # 뭐임??
        
        # Account for [CLS] and [SEP]
        special_tokens_count = 2
        if len(tokens) > args.max_seq_len - special_tokens_count:
            tokens = tokens[: (args.max_seq_len - special_tokens_count)]
            slot_label_mask = slot_label_mask[:(args.max_seq_len - special_tokens_count)]

        # Add [SEP] token
        tokens += [sep_token]
        token_type_ids = [sequence_a_segment_id] * len(tokens)
        slot_label_mask += [pad_token_label_id]

        # print("=====================")
        # print(all_input_tokens)

        # Add [CLS] token
        tokens = [cls_token] + tokens
        token_type_ids = [cls_token_segment_id] + token_type_ids
        slot_label_mask = [pad_token_label_id] + slot_label_mask

        # print("=====================")
        # print(all_input_tokens)

        input_ids = tokenizer.convert_tokens_to_ids(tokens)

        # The mask has 1 for real tokens and 0 for padding tokens. Only real tokens are attended to.
        attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

        # Zero-pad up to the sequence length.
        padding_length = args.max_seq_len - len(input_ids)

        input_ids = input_ids + ([pad_token_id] * padding_length)

        attention_mask = attention_mask + ([0 if mask_padding_with_zero else 1] * padding_length)
        token_type_ids = token_type_ids + ([pad_token_segment_id] * padding_length)
        slot_label_mask = slot_label_mask + ([pad_token_label_id] * padding_length)

        all_input_ids.append(input_ids)
        all_attention_mask.append(attention_mask)
        all_token_type_ids.append(token_type_ids)
        all_slot_label_mask.append(slot_label_mask)

    # Change to Tensor
    all_input_ids = torch.tensor(all_input_ids, dtype=torch.long)
    all_attention_mask = torch.tensor(all_attention_mask, dtype=torch.long)
    all_token_type_ids = torch.tensor(all_token_type_ids, dtype=torch.long)
    all_slot_label_mask = torch.tensor(all_slot_label_mask, dtype=torch.long)

    dataset = TensorDataset(all_input_ids, all_attention_mask, all_token_type_ids, all_slot_label_mask)

    # print("=====================")
    # print(all_input_tokens)
    # quit()

    return dataset, all_input_tokens

def get_device():
    return "cuda" if torch.cuda.is_available() else "cpu"


if __name__ == "__main__":

    print(os.getcwd())

    MODEL_DIR = './model' # edit configuration에서 working directory 확인하기 (파이참에서 상대 경로 쓸 경우)

    device = get_device()

    
    # modify ELECTRAConfig


    model = AutoModelForTokenClassification.from_pretrained(MODEL_DIR, torchscript=True)  # Config will be automatically loaded from model_dir
    model.to(device)
    model.eval()


    # prepare input
    training_params = get_args(MODEL_DIR)

    args = training_params['training_args']
    label_lst = training_params['label_lst']

    # Convert input file to TensorDataset
    pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
    tokenizer = load_tokenizer(args)
    lines = read_input_file('./sample.txt')
    dataset, all_input_tokens = convert_input_file_to_tensor_dataset(lines, args, tokenizer, pad_token_label_id)


    sampler = SequentialSampler(dataset)
    data_loader = DataLoader(dataset, sampler=sampler, batch_size=32)

    for batch in tqdm(data_loader, desc="predicting"):

        batch = tuple(t.to(device) for t in batch)

        with torch.no_grad():
            inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": None} # label이 None이므로 
            if args.model_type != "distilkobert":
                inputs["token_type_ids"] = batch[2]

            # pointer 같은 게 아니고... 파라미터 명을 같이 보낼 수 있다 

            # outputs = model(**inputs)
            traced_script = torch.jit.trace(model, inputs["input_ids"])
            # traced_script.save('./electra_traced.pt')
            # traced_model = torch.jit.trace(model, (inputs))
            # torch.jit.save(model, 'electra_script.pt')
            print(model)
            quit()





    


    
