NUM_GPUS=8
output_path=/path/to/save_dir
txt_file=/path/to/test_file_list.txt
triplane_path=/path/to/triplane_path
latent_root=/path/to/latent_root
ms_feature_root=/path/to/ms_feature_root
num_samples=10000

MODEL_FLAGS="--learn_sigma True --uncond_p 0.2 --image_size 128 --finetune_decoder True --diffusion_steps 1000 --predict_xstart False --predict_type noise" 
TRAIN_FLAGS="--lr 1e-5 --batch_size 3 --schedule_sampler uniform --use_tv False --exp_name $output_path --log_interval 100 --save_interval 5000" 
DIFFUSION_FLAGS="--noise_schedule cosine_light"
SAMPLE_FLAGS="--num_samples 10 --sample_c 1.0 "
DATASET_FLAGS="--data_dir $triplane_path --start_idx 0 --end_idx $num_samples  --mode triplane --txt_file $txt_file --latent_root $latent_root --ms_feature_root $ms_feature_root"
mpiexec -n $NUM_GPUS python base_train.py $MODEL_FLAGS  $TRAIN_FLAGS $SAMPLE_FLAGS $DIFFUSION_FLAGS $DATASET_FLAGS
