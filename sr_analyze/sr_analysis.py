import torch
import progressbar
import pandas as pd
from transformers import GPT2Tokenizer, GPT2LMHeadModel

class Args():
    def __init__(self):
        self.model_name = 'gpt2-xl'
        self.decoding_method = 'topp'
        self.prefix_len = 32
        self.decoding_len = 128
        self.data_path = '../dataset/webtext.json'
args = Args()

torch.cuda.set_device('cuda:6')

device = 'cuda:6'
cuda_available = torch.cuda.is_available()

model_name = 'gpt2-xl'
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)
model.to(device)
model.eval()

print ('Loading data...')
from dataclass import Data

data = Data(tokenizer, args.prefix_len, args.decoding_len, args.data_path)
print ('Data loaded.')


dict = {'sr_1': [], 'sr_2': [], 'sr_3': [], 'sr_4': [], 
        'sr_t2':[], 'sr_t4':[], 'sr_t6':[], 'sr_t8':[], 'sr_t10':[],
        'end_2':[], 'end_4':[], 'end_6':[], 'end_8':[], 'end_10':[]}

def inference_one_instance(args, data, index, model, cuda_available, device):
    decoding_method = args.decoding_method
    
    input_ids = data.prefix_token_id_list[index]
    input_ids = torch.LongTensor(input_ids).view(1,-1)
    _, prefix_len = input_ids.size()
    if cuda_available:
        input_ids = input_ids.cuda(device)

    decoding_len = args.decoding_len
    all_output_text_list = []
    
    with torch.no_grad():
        if decoding_method == 'greedy':
            output = model.generate(input_ids, do_sample=False,  max_length=32 + 128,  pad_token_id=50256, eos_token_id=50256)
        elif decoding_method == 'topk':
            output = model.generate(input_ids, do_sample=True,  max_length=32 + 128, top_k=5,  pad_token_id=50256, eos_token_id=50256)
        else:
            output = model.generate(input_ids, do_sample=True,  max_length=32 + 128, top_p=0.95,  pad_token_id=50256, eos_token_id=50256)

        dict['sr_1'].append(output[1].item())
        dict['sr_2'].append(output[2].item())
        dict['sr_3'].append(output[3].item())
        dict['sr_4'].append(output[4].item())
        dict['sr_t2'].append(output[5].squeeze(0)[0].item())
        dict['sr_t4'].append(output[5].squeeze(0)[1].item())
        dict['sr_t6'].append(output[5].squeeze(0)[2].item())
        dict['sr_t8'].append(output[5].squeeze(0)[3].item())
        dict['sr_t10'].append(output[5].squeeze(0)[4].item())
        dict['end_2'].append(output[6][0].item())
        dict['end_4'].append(output[6][1].item())
        dict['end_6'].append(output[6][2].item())
        dict['end_8'].append(output[6][3].item())
        dict['end_10'].append(output[6][4].item())

print ('Performing inference...')
data_num = min(1000, len(data.prefix_token_id_list))
print (data_num)
p = progressbar.ProgressBar(data_num)
p.start()
result_list = []
with torch.no_grad():
    for index in range(data_num):
        p.update(index)
        one_res_dict = inference_one_instance(args, data, index, model, cuda_available, device)
        result_list.append(one_res_dict)
p.finish()
print ('Inference completed!')

print(dict)
df = pd.DataFrame(dict)
df.to_csv('./sr_analysis' + args.decoding_method + '.csv', index=False)
    
