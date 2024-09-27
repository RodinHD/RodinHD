# RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models [ECCV 2024]

By [Bowen Zhang](http://home.ustc.edu.cn/~zhangbowen)\*, [Yiji Cheng](https://www.linkedin.com/in/yiji-cheng-a8b922213/?originalSubdomain=cn)\*, [Chunyu Wang](https://www.chunyuwang.org/)†, [Ting Zhang](https://hellozting.github.io/), [Jiaolong Yang](https://jlyang.org/), [Yansong Tang](https://andytang15.github.io/), [Feng Zhao](https://en.auto.ustc.edu.cn/2021/0616/c26828a513169/page.htm), [Dong Chen](http://www.dongchen.pro/), and [Baining Guo](https://www.microsoft.com/en-us/research/people/bainguo/).

[Paper](http://arxiv.org/abs/2407.06938) | [Project Page](https://rodinhd.github.io/) | [Code](https://github.com/RodinHD/RodinHD)

## Environment Setup
We recommend using Anaconda to create a new environment and install the dependencies. Our code is tested with Python 3.8 on Linux. Our model is trained and can be inferred using NVIDIA V100 GPUs.
```
conda env create -n rodinhd python=3.8
conda activate rodinhd
pip install -r requirements.txt
```

## Data Preparation

Due to organization policy, the training data is not publicly available. You can prepare your own data following the instructions below. Your 3D dataset can be organized as follows:
```
data
├── obj_00
│   ├── img_proc_fg_000000.png
│   ├── img_proc_fg_000001.png
│   ├── ...
|   ├── metadata_000000.json
|   ├── metadata_000001.json
|   ├── ...
├── obj_01
|   ├── ...
```

Then encode the multi-scale vae features of the frontal images of each object using the following command:
```bash
cd scripts
python encode_multiscale_feature.py --root /path/to/data --output_dir /path/to/feature --txt_file /path/to/txt_file --start_idx 0 --end_idx 1000
``` 
Where `--txt_file` is a txt file containing the list of objects to be encoded, and can be obtained by `ls /path/to/data > /path/to/txt_file`.

## Inference

Inference the base diffusion model:
```bash
cd scripts
sh base_sample.sh
```

Then inference the upsample diffusion model:
```bash
cd scripts
sh upsample_sample.sh
```

You need to modify the arguments in the scripts to fit your own data path.

## Training

### Triplane Fitting

We first fit the shared feature decoder with the proposed task-replay and identity-aware weight consolidation strategies using:
```bash
cd Renderer
sh fit_stage1.sh
```

Then we fix the shared feature decoder and fit each triplane per object using:
```bash
sh fit_stage2.sh
```

You need to modify the arguments in the scripts to fit your own data path.

### Triplane Diffusion

After fitting the triplanes, we train the diffusion model using:
```bash
sh ../scripts/base_train.sh
```

Then we train the upsample diffusion model using:
```bash
sh ../scripts/upsample_train.sh
```

You need to modify the arguments in the scripts to fit your own data path.

## Acknowledgement

This repository is built upon [improved-diffusion](https://github.com/openai/improved-diffusion), [torch-ngp](https://github.com/ashawkey/torch-ngp) and [Rodin](https://3d-avatar-diffusion.microsoft.com/). Thanks for their great work!

## Citation

If you find our work useful in your research, please consider citing:

```bibtex
@article{zhang2024rodinhd,
  title={RodinHD: High-Fidelity 3D Avatar Generation with Diffusion Models},
  author={Zhang, Bowen and Cheng, Yiji and Wang, Chunyu and Zhang, Ting and Yang, Jiaolong and Tang, Yansong and Zhao, Feng and Chen, Dong and Guo, Baining},
  journal={arXiv preprint arXiv:2407.06938},
  year={2024}
}
```

