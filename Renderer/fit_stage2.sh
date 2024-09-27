export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
fitting_obj_list=/path/to/fitting_obj_list.txt
save_dir=/path/to/save_triplane
ckpt=/path/to/stage1/ckpt
data_root=/path/to/data
rm -rf ${save_dir}
python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O --start_idx 1000 --end_idx 1001 \
    --bound 1.0 --scale 0.6 --dt_gamma 0 \
    --triplane_channels 32 \
    --data_root ${data_root} \
    --ckpt ${ckpt} \
    --iters 30000  --lr1 0 --eval_freq 10 
