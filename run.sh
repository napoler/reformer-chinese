# python3 train.py --raw --epochs 2 --min_length 64 --batch_size 10 --stride 768 --num_pieces 10 --output_dir /kaggle/model/ --pretrained_model /kaggle/working/model/  --tokenizer_path /kaggle/working/model/vocab.txt --model_config /kaggle/working/model/config.json


# python3 train.py --epochs 2 --device cpu --batch_size 4 --gradient_accumulation 2 --lr 5e-05
python3 train.py --epochs 1 --device cpu --batch_size 640 --gradient_accumulation 1 --lr 0.01 --num_pieces 10 --dim 64  --depth 6  --full_attn_thres 128 --stride 32

#33
python3 train.py --epochs 1 --device cpu --batch_size 320 --gradient_accumulation 1 --lr 0.01 --num_pieces 10 --depth 6  --full_attn_thres 128 --dim 128  --stride 60 --pretrained_model  model/

# 训练根据关键词造句
# python3 train_sent.py --raw --epochs 2 --device cpu --batch_size 1 --gradient_accumulation 1 --lr 0.01 --num_pieces 1
python3 train_sent.py --output_dir model --batch_size 4 --epochs 4 --device cuda

