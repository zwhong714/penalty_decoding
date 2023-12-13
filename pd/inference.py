# coding=utf-8
from time import time
import torch
from tqdm import tqdm
import torch.nn as nn
import torch.nn.functional as F
import torch.multiprocessing as mp
import argparse, os
import numpy as np
import time
import sys
sys.path.append('../../../')

from transformers import GPT2Tokenizer, GPT2LMHeadModel


def inference_batch_instance(batch_size, args, data, index, eos_token_id, model, cuda_available, device, tokenizer):
    decoding_method = args.decoding_method

    input_ids_all = []
    for i in range(index, index + batch_size):
        input_ids_all.append(data.prefix_token_id_list[i])

    input_ids = torch.tensor(input_ids_all).to('cuda')
   

    all_output_text_list = []
    with torch.no_grad():
        if decoding_method == 'greedy':
            output = model.generate(input_ids, do_sample=False, pad_token_id=50256, eos_token_id=50256, max_length=128 + 32)
            output_text = tokenizer.batch_decode(output[:, args.prefix_len:], skip_special_tokens=True)
            all_output_text_list = output_text
        elif decoding_method == 'beam':
            output = model.generate(input_ids, do_sample=False, pad_token_id=50256, eos_token_id=50256, max_length=128 + 32, num_beams=args.beam)
            output_text = tokenizer.batch_decode(output[:, args.prefix_len:], skip_special_tokens=True)
            all_output_text_list = output_text
        elif decoding_method == 'contrastive':
            output = model.generate(input_ids, do_sample=False, pad_token_id=50256, eos_token_id=50256, max_length=128 + 32, top_k=args.topk, penalty_alpha=args.penalty_alpha)
            output_text = tokenizer.batch_decode(output[:, args.prefix_len:], skip_special_tokens=True)
            all_output_text_list = output_text
        elif decoding_method == 'topk':
            output = model.generate(input_ids, do_sample=True, pad_token_id=50256, eos_token_id=50256, max_length=128 + 32, top_k=args.topk)
            output_text = tokenizer.batch_decode(output[:, args.prefix_len:], skip_special_tokens=True)
            all_output_text_list = output_text
        elif decoding_method == 'topp':
            output = model.generate(input_ids, do_sample=True, pad_token_id=50256, eos_token_id=50256, max_length=128 + 32, top_p=args.topp)
            output_text = tokenizer.batch_decode(output[:, args.prefix_len:], skip_special_tokens=True)
            all_output_text_list = output_text
        elif decoding_method == 'near_greedy':
            output = model.generate(input_ids, do_sample=False, pad_token_id=50256, eos_token_id=50256, max_length=128 + 32, repetition_penalty=args.penalty_alpha)
            output_text = tokenizer.batch_decode(output[:, args.prefix_len:], skip_special_tokens=True)
            all_output_text_list = output_text
        elif decoding_method == 'typical':
            output = model.generate(input_ids, do_sample=True, pad_token_id=50256, eos_token_id=50256, max_length=128 + 32, typical_p=args.typical)
            output_text = tokenizer.batch_decode(output[:, args.prefix_len:], skip_special_tokens=True)
            all_output_text_list = output_text
        elif decoding_method == 'eta':
            output = model.generate(input_ids, do_sample=True, pad_token_id=50256, eos_token_id=50256, max_length=128 + 32, eta_cutoff=args.eta)
            output_text = tokenizer.batch_decode(output[:, args.prefix_len:], skip_special_tokens=True)
            all_output_text_list = output_text

    res_dict = []

    for i in range(len(all_output_text_list)):
        res_dict.append({'prefix_text': data.prefix_text_list[index + i], 
                         'reference_text': data.reference_text_list[index + i],
                         'generated_result': all_output_text_list[i]})


    return res_dict




def parse_config():
    parser = argparse.ArgumentParser()
    # model and data configuration
    parser.add_argument("--model_name", type=str)
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--data_name", type=str)
    # decoding configuration
    parser.add_argument("--decoding_method", type=str)
    parser.add_argument("--prefix_len", type=int)
    parser.add_argument("--decoding_len", type=int)
    parser.add_argument("--number_of_instance_to_generate_per_method", type=int, default=1)
    # save configuration
    parser.add_argument("--save_path_prefix", type=str)
    parser.add_argument("--resistance_function", type=str)
    parser.add_argument("--topk", type=int)
    parser.add_argument("--topp", type=float)
    parser.add_argument("--typical", type=float)
    parser.add_argument("--eta", type=float)
    parser.add_argument('--num', type=int, default=500)
    parser.add_argument('--beam', type=int, default=5)
    parser.add_argument("--penalty_alpha", type=float, default=0.8)
    return parser.parse_args()




if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    args = parse_config()
    device = torch.device('cuda')

    save_path_prefix = args.save_path_prefix + '{}/{}/{}/'.format(args.model_name, args.data_name, args.decoding_method)
    import os
    if os.path.exists(save_path_prefix):
        pass
    else: # recursively construct directory
        os.makedirs(save_path_prefix, exist_ok=True)
    
    # save_name = '{}_result_{}.json'.format(args.decoding_method, args.resistance_function)
    if args.decoding_method in ['contrastive', 'resistance', 'topk']:
        save_name = '{}_result_{}.json'.format(args.decoding_method, args.topk)
        # save_name = '{}_result.json'.format(args.decoding_method)
    elif args.decoding_method in ['nucleus']:
        save_name = '{}_result_{}.json'.format(args.decoding_method, args.topp)
    else:
        save_name = '{}_result.json'.format(args.decoding_method)
    
    print(f'[!] save name is: {save_name}')
    save_path = save_path_prefix + save_name
    print ('Result saving path is {}'.format(save_path))



    print ('Loading model...')
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    model.eval()
    tokenizer = GPT2Tokenizer.from_pretrained(args.model_name)

    eos_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "left"
    tokenizer.pad_token = tokenizer.eos_token
    
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded.')



    print ('Loading data...')
    from dataclass import Data
    data = Data(tokenizer, args.prefix_len, args.decoding_len, args.data_path)
    print ('Data loaded.')


    print ('---------------Performing inference-------------------')
    data_num = len(data.prefix_token_id_list)
    data_num = min(data_num, args.num)
    print (data_num)
    result_list = []
    batch_size = 50
    with torch.no_grad():
        for index in tqdm(range(0, data_num, batch_size)):

            batch_res_dict = inference_batch_instance(batch_size, args, data, index, eos_token_id, model, cuda_available, device, tokenizer)
            result_list.extend(batch_res_dict)
    print ('-----------------Inference completed--------------------')

    print('--------------------File IO-----------------------------')
    import json
    with open(save_path, 'w') as outfile:
        json.dump(result_list, outfile, indent=4)