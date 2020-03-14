# python3 train.py --raw --epochs 2 --min_length 64 --batch_size 10 --stride 768 --num_pieces 10 --output_dir /kaggle/model/ --pretrained_model /kaggle/working/model/  --tokenizer_path /kaggle/working/model/vocab.txt --model_config /kaggle/working/model/config.json


# python3 train.py --epochs 2 --device cpu --batch_size 4 --gradient_accumulation 2 --lr 5e-05
python3 train.py --epochs 2 --device cpu --batch_size 4 --gradient_accumulation 2 --lr 0.01 --num_pieces 100