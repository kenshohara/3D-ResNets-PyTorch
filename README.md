# 3D ResNets for Action Recognition

## Update (2018/2/21)

Our paper "Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?" is accepted to CVPR2018!  
We update the paper information.

## Update (2018/01/16)

We uploaded some of fine-tuned models on UCF-101 and HMDB-51.

* ResNeXt-101 fine-tuned on UCF-101 (split1)
* ResNeXt-101 (64 frame inputs) fine-tuned on UCF-101 (split1)
* ResNeXt-101 fine-tuned on HMDB-51 (split1)
* ResNeXt-101 (64 frame inputs) fine-tuned on HMDB-51 (split1)

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
Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 6546-6555, 2018.
](http://openaccess.thecvf.com/content_cvpr_2018/html/Hara_Can_Spatiotemporal_3D_CVPR_2018_paper.html)

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

```bibtex
@inproceedings{hara3dcnns,
  author={Kensho Hara and Hirokatsu Kataoka and Yutaka Satoh},
  title={Can Spatiotemporal 3D CNNs Retrace the History of 2D CNNs and ImageNet?},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={6546--6555},
  year={2018},
}
```

## Pre-trained models

Pre-trained models are available [here](https://drive.google.com/drive/folders/1zvl89AgFAApbH0At-gMuZSeQB_LpNP-M?usp=sharing).  
All models are trained on Kinetics.  
ResNeXt-101 achieved the best performance in our experiments. (See [paper](https://arxiv.org/abs/1711.09577) in details.)

```misc
resnet-18-kinetics.pth: --model resnet --model_depth 18 --resnet_shortcut A
resnet-34-kinetics.pth: --model resnet --model_depth 34 --resnet_shortcut A
resnet-34-kinetics-cpu.pth: CPU ver. of resnet-34-kinetics.pth
resnet-50-kinetics.pth: --model resnet --model_depth 50 --resnet_shortcut B
resnet-101-kinetics.pth: --model resnet --model_depth 101 --resnet_shortcut B
resnet-152-kinetics.pth: --model resnet --model_depth 152 --resnet_shortcut B
resnet-200-kinetics.pth: --model resnet --model_depth 200 --resnet_shortcut B
preresnet-200-kinetics.pth: --model preresnet --model_depth 200 --resnet_shortcut B
wideresnet-50-kinetics.pth: --model wideresnet --model_depth 50 --resnet_shortcut B --wide_resnet_k 2
resnext-101-kinetics.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
densenet-121-kinetics.pth: --model densenet --model_depth 121
densenet-201-kinetics.pth: --model densenet --model_depth 201
```

Some of fine-tuned models on UCF-101 and HMDB-51 (split 1) are also available.

```misc
resnext-101-kinetics-ucf101_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
resnext-101-64f-kinetics-ucf101_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64
resnext-101-kinetics-hmdb51_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32
resnext-101-64f-kinetics-hmdb51_split1.pth: --model resnext --model_depth 101 --resnet_shortcut B --resnext_cardinality 32 --sample_duration 64
```

### Performance of the models on Kinetics

This table shows the averaged accuracies over top-1 and top-5 on Kinetics.

| Method | Accuracies |
|:---|:---:|
| ResNet-18 | 66.1 |
| ResNet-34 | 71.0 |
| ResNet-50 | 72.2 |
| ResNet-101 | 73.3 |
| ResNet-152 | 73.7 |
| ResNet-200 | 73.7 |
| ResNet-200 (pre-act) | 73.4 |
| Wide ResNet-50 | 74.7 |
| ResNeXt-101 | 75.4 |
| DenseNet-121 | 70.8 |
| DenseNet-201 | 72.3 |

## Requirements

* [PyTorch](http://pytorch.org/)

```bash
conda install pytorch torchvision cuda80 -c soumith
```

* FFmpeg, FFprobe

```bash
wget http://johnvansickle.com/ffmpeg/releases/ffmpeg-release-64bit-static.tar.xz
tar xvf ffmpeg-release-64bit-static.tar.xz
cd ./ffmpeg-3.3.3-64bit-static/; sudo cp ffmpeg ffprobe /usr/local/bin;
```

* Python 3

## Preparation

### ActivityNet

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler).
* Convert from avi to jpg files using ```utils/video_jpg.py```

```bash
python utils/video_jpg.py avi_video_directory jpg_video_directory
```

* Generate fps files using ```utils/fps.py```

```bash
python utils/fps.py avi_video_directory jpg_video_directory
```

### Kinetics

* Download videos using [the official crawler](https://github.com/activitynet/ActivityNet/tree/master/Crawler/Kinetics).
  * Locate test set in ```video_directory/test```.
* Convert from avi to jpg files using ```utils/video_jpg_kinetics.py```

```bash
python utils/video_jpg_kinetics.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_kinetics.py```

```bash
python utils/n_frames_kinetics.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/kinetics_json.py```
  * The CSV files (kinetics_{train, val, test}.csv) are included in the crawler.

```bash
python utils/kinetics_json.py train_csv_path val_csv_path test_csv_path dst_json_path
```

### UCF-101

* Download videos and train/test splits [here](http://crcv.ucf.edu/data/UCF101.php).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/ucf101_json.py```
  * ```annotation_dir_path``` includes classInd.txt, trainlist0{1, 2, 3}.txt, testlist0{1, 2, 3}.txt

```bash
python utils/ucf101_json.py annotation_dir_path
```

### HMDB-51

* Download videos and train/test splits [here](http://serre-lab.clps.brown.edu/resource/hmdb-a-large-human-motion-database/).
* Convert from avi to jpg files using ```utils/video_jpg_ucf101_hmdb51.py```

```bash
python utils/video_jpg_ucf101_hmdb51.py avi_video_directory jpg_video_directory
```

* Generate n_frames files using ```utils/n_frames_ucf101_hmdb51.py```

```bash
python utils/n_frames_ucf101_hmdb51.py jpg_video_directory
```

* Generate annotation file in json format similar to ActivityNet using ```utils/hmdb51_json.py```
  * ```annotation_dir_path``` includes brush_hair_test_split1.txt, ...

```bash
python utils/hmdb51_json.py annotation_dir_path
```

## Running the code

Assume the structure of data directories is the following:

```misc
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

```bash
python main.lua -h
```

Train ResNets-34 on the Kinetics dataset (400 classes) with 4 CPU threads (for data loading).  
Batch size is 128.  
Save models at every 5 epochs.
All GPUs is used for the training.
If you want a part of GPUs, use ```CUDA_VISIBLE_DEVICES=...```.

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --model resnet \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Continue Training from epoch 101. (~/data/results/save_100.pth is loaded.)

```bash
python main.py --root_path ~/data --video_path kinetics_videos/jpg --annotation_path kinetics.json \
--result_path results --dataset kinetics --resume_path results/save_100.pth \
--model_depth 34 --n_classes 400 --batch_size 128 --n_threads 4 --checkpoint 5
```

Fine-tuning conv5_x and fc layers of a pretrained model (~/data/models/resnet-34-kinetics.pth) on UCF-101.

```bash
python main.py --root_path ~/data --video_path ucf101_videos/jpg --annotation_path ucf101_01.json \
--result_path results --dataset ucf101 --n_classes 400 --n_finetune_classes 101 \
--pretrain_path models/resnet-34-kinetics.pth --ft_begin_index 4 \
--model resnet --model_depth 34 --resnet_shortcut A --batch_size 128 --n_threads 4 --checkpoint 5
```
