export PATH=/usr/local/cuda/bin:~/.local/bin:$PATH
save_dir=/path/to/save_triplane_and_mlp
data_root=/path/to/data
fitting_obj_list=/path/to/fitting_obj_list.txt
rm -rf ${save_dir}
mkdir -p ${save_dir}
python main.py \
    ${fitting_obj_list} \
    ${save_dir} \
    --workspace ${save_dir} \
    -O \
    --start_idx 0 --end_idx 256 \
    --bound 1.0 --scale 0.6 --dt_gamma 0 \
    --triplane_channels 32 \
    --ckpt "scratch" \
    --data_root ${data_root} \
    --out_loop_eps 30 --iters 6000 --lr0 2e-2 --lr1 2e-3 --eval_freq 5 \
    --l1_reg_weight 1e-6 \
    --tv_weight 2e-4 \
    --dist_weight 2e-5 \
    --iwc_weight 0.1
