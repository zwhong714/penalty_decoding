import json
import torch
import argparse
import progressbar
from transformers import AutoTokenizer, GPT2LMHeadModel
from tqdm import tqdm
import ipdb

class Data:
    def __init__(self, tokenizer, test_path, data_path, prefix_len, decoding_len):
        self.tokenizer = tokenizer
        self.prefix_len = prefix_len
        self.decoding_len = decoding_len
        self.prefix_token_id_list = self.get_prefix_file(data_path)
        self.result_token_id_list = self.get_result_file(test_path)
        print('Evaluation number is {}'.format(len(self.prefix_token_id_list)))
    
    def get_prefix_file(self, data_path):
        print ('Get prefix from {}'.format(data_path))
        prefix_token_id_list = []
        
        import json
        with open(data_path) as f:
            data = json.load(f)

        n = len(data)
        print ('Prefix number is {}'.format(n))
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            item = data[i]
            text = item['prefix_text']
            token_id_list = self.get_one_prefix(text)
            if token_id_list != []:
                prefix_token_id_list.append(token_id_list)
        
        return prefix_token_id_list
    
    def get_one_prefix(self, text):
        tokens = self.tokenizer.tokenize(text)

        token_id_list = self.tokenizer.convert_tokens_to_ids(tokens)
        prefix_id_list = token_id_list[:self.prefix_len]
        return prefix_id_list

    def get_result_file(self, test_path):
        print ('Get result from {}'.format(test_path))
        result_token_id_list = []

        data = json.load(open(test_path))
        n = len(data)
        print ('Result number is {}'.format(n))
        
        assert n == len(self.prefix_token_id_list)
        p = progressbar.ProgressBar(n)
        p.start()
        for i in range(n):
            p.update(i)
            temp_token_id_list = []
            
            temp_token_id_list.append(self.get_one_result(data[i]['generated_result']))

            result_token_id_list.append(temp_token_id_list)
        p.finish()
        return result_token_id_list

    def get_one_result(self, text):
        result_tokens = self.tokenizer.tokenize(text)
        result_id_list = self.tokenizer.convert_tokens_to_ids(result_tokens)

        return result_id_list



def inference_one_instance(args, prefix_token_id_list, source_token_id_list, model, cuda_available, device):
    input_ids = torch.LongTensor(prefix_token_id_list + source_token_id_list).view(1, -1)

    if cuda_available:
        input_ids = input_ids.cuda(device)
    
    with torch.no_grad():
        
        output = model(input_ids=input_ids)
        greedy = torch.max(output.logits, dim=-1).indices[0][len(prefix_token_id_list) - 1: -1]
        
        target = input_ids[0][len(prefix_token_id_list):]
        
        
        if target.shape[0] == 0:
            return 1.0 

        ratio = round(torch.mean(torch.eq(greedy, target).float()).item(), 4)
        
        return ratio


def get_config(decoding_method):
    prefix_len, decoding_len = 32, 128
    
    return prefix_len, decoding_len


def parse_config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--decoding_method", type=str, default='topp')
    parser.add_argument("--data_path", type=str, default='../../inference_results/gpt2-xl/webtext/near_greedy/near_greedy_result.json')
    parser.add_argument("--data_name", type=str, default='webtext')
    parser.add_argument("--model_name", type=str, default='gpt2-xl')
    return parser.parse_args()


if __name__ == '__main__':
    if torch.cuda.is_available():
        print ('Cuda is available.')
    cuda_available = torch.cuda.is_available()
    device = torch.device('cuda:2')

    args = parse_config()
    
    save_path = f"../../inference_results/{args.model_name}/{args.data_name}/{args.decoding_method}/{args.decoding_method}_greedy_ratio_result.json"

    test_path = args.data_path


    print ('evaluation save name is {}'.format(save_path))
    print('evaluation file name is', test_path)

    print ('Model loading...')
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = GPT2LMHeadModel.from_pretrained(args.model_name)
    if cuda_available:
        model = model.to(device)
    model.eval()
    print ('Model loaded')

    print ('Data loading...')
    prefix_len, decoding_len = get_config(args.decoding_method)
    data = Data(tokenizer, test_path, args.data_path, prefix_len, decoding_len)
    print ('Data loaded')


    ######################################################
    print ('Computing greedy ratio...')
    greedy_ratio = 0.0
    with torch.no_grad():
        greddy_same_count = 0
        for idx in tqdm(range(len(data.prefix_token_id_list))):
            
            
            greedy_ratio += inference_one_instance(args, data.prefix_token_id_list[idx], data.result_token_id_list[idx][0], model, cuda_available, device)
        
        greedy_ratio = greedy_ratio / (len(data.prefix_token_id_list))
        print ('Greedy ratio calculated completed')
        print ('greedy ratio: ', greedy_ratio)
    
    import json
    with open(save_path, 'w') as outfile:
        json.dump({'greedy_ratio': greedy_ratio, 'test_cases': len(data.prefix_token_id_list)}, outfile, indent=4)

    
