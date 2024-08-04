# PANO-ECHO 
code for PANO-ECHO: PANOramic depth prediction enhancement with ECHO features.
This repository is a fork, a changed version of the original panoformer github repository *[PanoFormer](https://github.com/zhijieshen-bjtu/PanoFormer)*
## 0.Genration of 360° equirectangular dataset from SoundSpace
0.Download the SoundSpace dataset (Matterport3d and Replica) from *[SoundSpace](https://github.com/facebookresearch/)*
1.Install habitat-sim: Please refer to *[habitat-sim](https://github.com/facebookresearch/habitat-sim)* for the installation of habitat-sim.
2.Render equirectangular RGB and depth observations:
```bash
cd prepare_datasets
bash img360_extraction.sh
bash img360_organisation.sh
bash img360_split_dataset.sh
```
The final generated datasets are named dataset_realEquirec_mp3d_organized and dataset_realEquirec_replica_organized

## 1.Training from scratch
```bash
cd PANO-ECHO
```
Train PANO-ECHO with echos on replica dataset
```bash
python train.py \
--disable_color_augmentation \
--disable_yaw_rotation_augmentation \
--disable_LR_filp_augmentation \
--exp_name exp_name \
--model PanoFormer \
--optimiser Adam \
--num_epochs 200 \
--mode train \
--dataset_use_ratio 1 \
--dataset replica \
--model_mode cross_attention \
--batch_size 2 \
```

Train PANO-ECHO with echos on Matterport3d dataset
```bash
python train.py \
--disable_color_augmentation \
--disable_yaw_rotation_augmentation \
--disable_LR_filp_augmentation \
--exp_name exp_name \
--model PanoFormer \
--optimiser Adam \
--num_epochs 200 \
--mode train \
--dataset_use_ratio 0.1 \
--dataset mp3d \
--model_mode cross_attention \
--batch_size 2 \
```

## 2.Test from trained model
Test PANO-ECHO with echos on replica dataset
```bash
python test.py \
--disable_color_augmentation \
--disable_yaw_rotation_augmentation \
--disable_LR_filp_augmentation \
--model PanoFormer \
--exp_name exp_name \
--mode test \
--dataset_use_ratio 1 \
--dataset replica \
--model_mode cross_attention \
--load_weights_dir path/to/exp_name/models/best
```

Test PANO-ECHO with echos on Matterport3d dataset
```bash
python test.py \
--disable_color_augmentation \
--disable_yaw_rotation_augmentation \
--disable_LR_filp_augmentation \
--model PanoFormer \
--exp_name exp_name \
--mode test \
--dataset_use_ratio 0.1 \
--dataset mp3d \
--model_mode cross_attention \
--load_weights_dir path/to/exp_name/models/best
```

For Unifuse and Bifuse, change the parameter --model to Unifuse and Bifuse respectively.
To train and test the baseline model, change --audio_enhanced to 0

## Acknowledgement
This project is based on *[PanoFormer](https://github.com/zhijieshen-bjtu/PanoFormer)*, *[Unifuse](https://github.com/alibaba/UniFuse-Unidirectional-Fusion)*, and *[Bifuse](https://github.com/yuhsuanyeh/BiFuse)*

The code for equirectangular dataset extraction is modified from [Dense 2D-3D Indoor Prediction with Sound via Aligned Cross-Modal Distillation (ICCV 2023)](https://github.com/HS-YN/DAPS/tree/main/DAPS) (MIT License).

If you find this repository useful, please consider citing:
@INPROCEEDINGS{10605546,
  author={Liu, Xiaohu and Brunetto, Amandine and Hornauer, Sascha and Moutarde, Fabien and Lu, Jialiang},
  booktitle={2024 IEEE Conference on Artificial Intelligence (CAI)}, 
  title={PANO-ECHO: PANOramic depth prediction enhancement with ECHO features}, 
  year={2024},
  volume={},
  number={},
  pages={1063-1070},
  keywords={Measurement;Visualization;Codes;Pipelines;Estimation;Predictive models;Distortion;Audio-Visual learning;panoramic depth estimation;multi-modal fusion},
  doi={10.1109/CAI59869.2024.00193}}
