bash config.sh
CUDA_VISIBLE_DEVICES=7 python ./inference.py\
    --model_name gpt2\
    --data_path /home/wenhongzhu/decdoing_algorithm/dataset/webtext.json\
    --data_name webtext\
    --decoding_method greedy\
    --prefix_len 32\
    --decoding_len 128\
    --save_path_prefix ./inference_results/

CUDA_VISIBLE_DEVICES=7 python ./metric/measure_diversity_mauve_gen_length.py\
    --test_path /home/wenhongzhu/decdoing_algorithm/MAIN_EXPERIENMENT/inference_results/gpt2/webtext/greedy/greedy_result.json
exit