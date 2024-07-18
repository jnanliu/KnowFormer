# KnowFormer: Revisiting Transformers for Knowledge Graph Reasoning

## 📋 Introduction
This is the implementation for the ICML 2024 Conference paper _KnowFormer: Revisiting Transformers for Knowledge Graph Reasoning_.

_Junnan Liu_, _Qianren Mao_\*, _Weifeng Jiang_, _Jianxin Li_

## 🚀 Getting Started

### Installation
You can create a conda virtual environment that can be used to run the project by using the following command.
```bash
conda create -n knowformer python=3.9
conda activate knowformer
pip install torch==2.1.0 torchvision==0.16.0 torchaudio==2.1.0 --index-url https://download.pytorch.org/whl/cu121
pip install torch_geometric==2.4.0 pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.1.0+cu121.html
pip install pytorch_lightning==1.9.1
pip install torchmetrics==0.11.4
pip install einops==0.7.0
```

### Usage
You can use the following commands to run KnowFormer. Please modify the argument `devices` based on your device.
<details>
<summary>FB15k-237</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \ 
               --strategy ddp 
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \ 
               --checkpoint_save_path ./experiments/fb15k-237/ \
               --data_path ./data/fb15k-237 \ 
               --batch_size 96 \
               --test_batch_size 96 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 32 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 0.5 \
               --remove_all \
               --num_negative_sample 8 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>WN18RR</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20  \
               --checkpoint_save_path ./experiments/train/wn18rr/ \
               --data_path ./data/wn18rr \
               --batch_size 32 \
               --test_batch_size 32 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 32 \
               --num_heads 4 \
               --loss_fn ce \
               --learning_rate 5e-3 \ 
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>NELL995</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/nell-995/ \
               --data_path ./data/nell-995 \
               --batch_size 16 \
               --test_batch_size 16 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 32 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 0.5 \
               --num_negative_sample 16 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>YAGO3-10</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 2 \
                --checkpoint_save_path ./experiments/train/yago3-10/ \
               --data_path ./data/yago3-10 \
               --batch_size 12 \
               --test_batch_size 12 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 32 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 1.0 \
               --num_negative_sample 16 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>FB15k-237v1</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/fb15k-237_v1/ \
               --data_path ./data/inductive/fb15k-237_v1 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 64 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 0.5 \
               --num_negative_sample 6 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>FB15k-237v2</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/fb15k-237_v2/ \
               --data_path ./data/inductive/fb15k-237_v2 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 64 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 0.5 \
               --num_negative_sample 6 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>FB15k-237v3</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/fb15k-237_v3/ \
               --data_path ./data/inductive/fb15k-237_v3 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 64 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 0.5 \
               --num_negative_sample 6 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>FB15k-237v4</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/fb15k-237_v4/ \
               --data_path ./data/inductive/fb15k-237_v4 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 64 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 0.5 \
               --num_negative_sample 6 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>WN18RRv1</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/wn18rr_v1/ \
               --data_path ./data/inductive/wn18rr_v1 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 32 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 0.5  \
               --num_negative_sample 8 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>WN18RRv2</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/wn18rr_v2/ \
               --data_path ./data/inductive/wn18rr_v2 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 32 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 0.5  \
               --num_negative_sample 8 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>WN18RRv3</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/wn18rr_v3/ \
               --data_path ./data/inductive/wn18rr_v3 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 32 \
               --num_heads 4 \
               --loss_fn ce \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>WN18RRv4</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/wn18rr_v4/ \
               --data_path ./data/inductive/wn18rr_v4 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 32 \
               --num_heads 4 \
               --loss_fn ce \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-4

```
</details>

<details>
<summary>NELL995v1</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/nell-995_v1/ \
               --data_path ./data/inductive/nell-995_v1 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 64 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 1.0  \
               --num_negative_sample 16 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-5

```
</details>

<details>
<summary>NELL995v2</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/nell-995_v2/ \
               --data_path ./data/inductive/nell-995_v2 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 64 \
               --num_heads 4 \
               --loss_fn ce \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-5

```
</details>

<details>
<summary>NELL995v3</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/nell-995_v3/ \
               --data_path ./data/inductive/nell-995_v3 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 64 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 1.0  \
               --num_negative_sample 16 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-5

```
</details>

<details>
<summary>NELL995v4</summary>

```bash
python main.py --seed 42 \
               --accelerator gpu \
               --strategy ddp \
               --precision 32 \
               --devices 4 \
               --max_epochs 20 \
               --checkpoint_save_path ./experiments/train/nell-995_v4/ \
               --data_path ./data/inductive/nell-995_v4 \
               --batch_size 64 \
               --test_batch_size 64 \
               --num_workers 8 \
               --num_layer 3 \
               --num_qk_layer 2 \
               --num_v_layer 3 \
               --hidden_dim 64 \
               --num_heads 4 \
               --loss_fn bce \
               --adversarial_temperature 1.0  \
               --num_negative_sample 16 \
               --learning_rate 5e-3 \
               --optimizer Adam \
               --weight_decay 1e-5

```
</details>

## 🎯 Acknowledgment
Our implementation is partially based on Project [NBFNet](https://github.com/KiddoZhu/NBFNet-PyG), and we appreciate their contributions.

## 🌟 Citation
If you used our work or found it helpful, please kindly cite our paper:
```
@inproceedings{liu2024knowformer,
  title={KnowFormer: Revisiting Transformers for Knowledge Graph Reasoning},
  author={Junnan Liu and Qianren Mao and Weifeng Jiang and Jianxin Li},
  booktitle={Forty-first International Conference on Machine Learning},
  year={2024}
}
```
