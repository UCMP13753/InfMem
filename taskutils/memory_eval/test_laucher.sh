export DATAROOT=../../hotpotqa

nohup python run.py \
    > logs/eval_infmem.log 2>&1 &
    
echo $! > run.pid