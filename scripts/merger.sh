CKPT=
BASE=

TARGET=$CKPT/huggingface
python3 scripts/model_merger.py \
    --backend "fsdp" \
    --hf_model_path $BASE \
    --local_dir $CKPT/actor \
    --target_dir $TARGET
cp $BASE/token*json $TARGET
cp $BASE/vocab.json $TARGET
# cp $BASE/chat_template.jinja $TARGET