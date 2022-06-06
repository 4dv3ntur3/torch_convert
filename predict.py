import os
import logging
import argparse
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, SequentialSampler
from transformers import AutoModelForTokenClassification

from utils import init_logger, load_tokenizer

import glob

logger = logging.getLogger(__name__)

def get_device(pred_config):
    return "cuda" if torch.cuda.is_available() and not pred_config.no_cuda else "cpu"

def get_args(pred_config):
    return torch.load(os.path.join(pred_config.model_dir, 'training_parameters.bin'))


def load_model(pred_config, args, device):
    # Check whether model exists
    # torch.load(training.bins)를 해서 학습할 당시 저장했던 model_dir를 찾도록 되어 있음
    # 이런 방식은 학습 이후 model 디렉토리 이름을 변경할 시 자꾸 path를 찾지 못하는 문제가 있음.
    # 그래서 pred_config.model_dir로 찾도록 한다.
    if not os.path.exists(pred_config.model_dir):
        raise Exception("Model doesn't exists! Train first!")

    try:
        # model = AutoModelForTokenClassification.from_pretrained(args.model_dir)  # Config will be automatically loaded from model_dir
        model = AutoModelForTokenClassification.from_pretrained(pred_config.model_dir)  # Config will be automatically loaded from model_dir
        model.to(device)
        model.eval()
        logger.info("***** Model Loaded *****")
    except:
        raise Exception("Some model files might be missing...")

    return model


def read_input_file(pred_config):
    lines = []
    with open(pred_config.input_file, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            words = line.split()
            lines.append(words)

    return lines


def convert_input_file_to_tensor_dataset(lines,
                                         pred_config,
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


def predict(pred_config):
    # load model and args
    training_params = get_args(pred_config)
    device = get_device(pred_config)

    args = training_params['training_args']
    label_lst = training_params['label_lst']

    model = load_model(pred_config, args, device)
    logger.info(args)
    logger.info(label_lst)

    file_list = []
    if pred_config.input_file is None: 
      ### load all files in sample_pred_in directory
      target = os.path.join(pred_config.input_dir, '*.txt')
      file_list = glob.glob(target)

    else:
      # not directory, for file
      file_list.append(pred_config.input_file)

    for input_file in file_list:

      pred_config.input_file = input_file

      # Convert input file to TensorDataset
      pad_token_label_id = torch.nn.CrossEntropyLoss().ignore_index
      tokenizer = load_tokenizer(args)
      lines = read_input_file(pred_config)
      dataset, all_input_tokens = convert_input_file_to_tensor_dataset(lines, pred_config, args, tokenizer, pad_token_label_id)

      # print(all_input_tokens)
      # quit()

      # Predict
      sampler = SequentialSampler(dataset)
      data_loader = DataLoader(dataset, sampler=sampler, batch_size=pred_config.batch_size)

      all_slot_label_mask = None
      preds = None
      sms = None
      for batch in tqdm(data_loader, desc="Predicting"):
          batch = tuple(t.to(device) for t in batch)

          with torch.no_grad():
              inputs = {"input_ids": batch[0],
                        "attention_mask": batch[1],
                        "labels": None} # label이 None이므로 
              if args.model_type != "distilkobert":
                  inputs["token_type_ids"] = batch[2]

              outputs = model(**inputs)

              # print(len(outputs)) # 1

              logits = outputs[0] # loss? 
              # print(logits.size()) # 3, 50, 28 (B, S, num_classes)
              # print(logits.detach().cpu().numpy())

              # logits = linear output
              sm = torch.nn.functional.softmax(logits, dim=-1)
              sm = sm.detach().cpu().numpy()
              # print(sm)
              # print(sm.size()) # 3, 50, 28

              if preds is None: # the very first one 
                  preds = logits.detach().cpu().numpy()
                  all_slot_label_mask = batch[3].detach().cpu().numpy()
                  sms = sm
              else:
                  preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
                  all_slot_label_mask = np.append(all_slot_label_mask, batch[3].detach().cpu().numpy(), axis=0)
                  sms = np.append(sms, axis=0)

      preds = np.argmax(preds, axis=2)


      slot_label_map = {i: label for i, label in enumerate(label_lst)}
      preds_list = [[] for _ in range(preds.shape[0])]

      sms_list = [[] for _ in range(sms.shape[0])]

      for i in range(preds.shape[0]):
          for j in range(preds.shape[1]):
              if all_slot_label_mask[i, j] != pad_token_label_id: # 일단 이 부분에서 뽑지 않으면 other도 해당되는 것 
                  preds_list[i].append(slot_label_map[preds[i][j]])
                  sms_list[i].append(sms[i][j])

      ######### 정답 후보 태그 3개까지 출력 
      sen_list = []
      for k in range(len(sms_list)):

        temp = sms_list[k] #19 (Number of tokens)

        # print(temp)
        # print(len(temp))
        # print("=======================")
        
        for tidx in range(len(temp)):
          
          tok_candidates = {}
          
          real_tok = all_input_tokens[k][tidx]
          tok = temp[tidx]
          ranked = np.argsort(tok)
          largest_indices = ranked[::-1][:3]

          # print("rank for", real_tok, " started")

          candids = {}
          for i in range(len(largest_indices)):
            idx = largest_indices[i]
            # # print(idx)
            # print(slot_label_map[idx], ": ", tok[idx])

            # 그냥 한 번에 볼 수 있게끔 수정 
            candids[i] = (slot_label_map[idx], tok[idx])
                  
          tok_candidates[real_tok] = [candids]
          sen_list.append(tok_candidates)

          # print("rank for", real_tok, " ended\n\n")

      for i in range(len(sen_list)):
        print(sen_list[i])

      # output file 경로가 따로 주어지지 않은 경우는 이렇게 
      if pred_config.output_file == None:
        ### modify output directory
        root = './sample_pred_out'
        model_dir = pred_config.model_dir.split('/')[-1]

        #### join()은 디렉토리의 구분 문자 (/, \\)가 들어 있으면 그것을 root로 본다
        save_dir = os.path.join(root, model_dir)
        save_fn = pred_config.input_file.split("/")[-1]

        if not os.path.exists(save_dir):
          os.makedirs(save_dir)

        pred_config.output_file = os.path.join(save_dir, save_fn)

      # Write to output file
      with open(pred_config.output_file, "w", encoding="utf-8") as f:
          for words, preds in zip(all_input_tokens, preds_list):
              line = ""
              for word, pred in zip(words, preds):
                  if pred == 'O':
                      line = line + word + " "
                  else:
                      line = line + "[{}:{}] ".format(word, pred)

              f.write("{}\n".format(line.strip()))

          pred_config.output_file = None

    logger.info("Prediction Done!")


if __name__ == "__main__":
    init_logger()
    parser = argparse.ArgumentParser()

    parser.add_argument("--input_file", default=None, type=str, help="Input file for prediction")
    parser.add_argument("--output_file", default=None, type=str, help="Output file for prediction")
    parser.add_argument("--model_dir", default="./model", type=str, help="Path to save, load model")

    parser.add_argument("--batch_size", default=32, type=int, help="Batch size for prediction")
    parser.add_argument("--no_cuda", action="store_true", help="Avoid using CUDA when available")

    parser.add_argument("--input_dir", default='./sample_pred_in', type=str, help="directory for input files")

    pred_config = parser.parse_args()
    predict(pred_config)