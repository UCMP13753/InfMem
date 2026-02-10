# if [ -z $(which aria2c) ]; then
#     sudo apt update
#     yes | sudo apt install aria2
# fi
# echo "https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json  
#     out=squad.json
# http://curtis.ml.cmu.edu/datasets/hotpot/hotpot_dev_distractor_v1.json  
#     out=hotpotqa_dev.json" > __download.txt
# aria2c -x 10 -s 10 -j 2 -i __download.txt
# rm __download.txt
mkdir downloaded_data
cd downloaded_data
wget https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_train.jsonl
wget https://huggingface.co/datasets/dgslibisey/MuSiQue/resolve/main/musique_ans_v1.0_dev.jsonl

wget -O squad_train.json https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json
wget -O squad_dev.json https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json

wget -O 2wiki_train.parquet https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/train.parquet
wget -O 2wiki_dev.parquet https://huggingface.co/datasets/xanhho/2WikiMultihopQA/resolve/main/dev.parquet

cd ../