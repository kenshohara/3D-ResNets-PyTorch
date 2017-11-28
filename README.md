# 3D ResNets for Action Recognition
## Update (2017/11/27)
We published [a new paper](https://arxiv.org/abs/1711.09577) on arXiv.  
We also added the following new models and their Kinetics pretrained models in this repository.  
* ResNet-50, 101, 152, 200
* Pre-activation ResNet-200
* Wide ResNet-50
* ResNeXt-101
* DenseNet-121, 201
In addition, we supported new datasets (UCF-101 and HDMB-51) and fine-tuning functions.

Some minor changes are included.
* Outputs are normalized by softmax in test.
  * If you do not want to perform the normalization, please use ```--no_softmax_in_test``` option.

## Summary
This is the PyTorch code for the following papers:

[
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?",  
arXiv preprint, arXiv:1711.09577, 2017.
](https://arxiv.org/abs/1711.09577)

[
Kensho Hara, Hirokatsu Kataoka, and Yutaka Satoh,  
"Learning Spatio-Temporal Features with 3D Residual Networks for Action Recognition",  
Proceedings of the ICCV Workshop on Action, Gesture, and Emotion Recognition, 2017.
](http://openaccess.thecvf.com/content_ICCV_2017_workshops/papers/w44/Hara_Learning_Spatio-Temporal_Features_ICCV_2017_paper.pdf)

This code includes training, fine-tuning and testing on Kinetics, ActivityNet, UCF-101, and HMDB-51.  
**If you want to classify your videos or extract video features of them using our pretrained models,
use [this code](https://github.com/kenshohara/video-classification-3d-cnn-pytorch).**

**The Torch (Lua) version of this code is available [here](https://github.com/kenshohara/3D-ResNets).**  
Note that the Torch version only includes ResNet-18, 34, 50, 101, and 152.

## Citation
If you use this code or pre-trained models, please cite the following:
```
@article{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  journal={arXiv preprint},
  volume={arXiv:1711.09577},
  year={2017},
}
```

## Pre-trained models
Pre-trained models are available at [releases](https://github.com/kenshohara/3D-ResNets-PyTorch/releases/tag/1.0).

## Requirements
* [PyTorch](http://pytorch.org/)
```
conda install pytorch torchvision cuda80 -c soumith
```
* FFmpeg, FFprobe
```
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```
* Python 3

## Preparation
### ActivityNet
* Download datasets using official crawler codes
* Convert from avi to jpg files using ```utils/video_jpg.py```
```
python utils/video_jpg.py avi_video_directory jpg_video_directory
```
* Generate fps files using ```utils/fps.py```
```
python utils/fps.py avi_video_directory jpg_video_directory
```

### Kinetics
* Download the Kinetics dataset using official crawler codes
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```utils/video_jpg_kinetics.py```
```
python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory
```
* Generate n_frames files using ```utils/n_frames_kinetics.py```
```
python utils/n_frames_kinetics.py jpg_video_directory
```
* Generate annotation file in json format similar to ActivityNet using ```utils/kinetics_json.py```
```
python utils/kinetics_json.py train_csv_path val_csv_path test_csv_path json_path
```

## Running the code
Assume the structure of data directories is the following:
```
~/
  data/
    kinetics_videos/
      jpg/
        .../ (directories of class names)
          .../ (directories of video names)
            ... (jpg files)
    results/
      save_100.pth
    kinetics.json
```

Confirm all options.
```
python main.lua -h
```

Train ResNets-34 on the Kinetics dataset (400 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.
```
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Continue Training from epoch 101. (~/data/results/save_100.pth is loaded.)
```
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_100.pth \
--batch_size 128 --n_threads 4 --checkpoint 5
```

Fine-tuning conv5_x and fc layers of a pretrained model (~/data/models/resnet-34-kinetics.pth) on UCF-101.
```
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 \
--batch_size 128 --n_threads 4 --checkpoint 5
```
