# Video Summarization with LSTM

This repository provides the data and implementation for video summarization with LSTM, i.e. vsLSTM and dppLSTM in our paper:

**[Video Summarization with Long Short-term Memory](http://www-scf.usc.edu/~zhan355/ke_eccv2016.pdf)**
<br>
[Ke Zhang](http://www-scf.usc.edu/~zhan355/index.html)\*, Wei-Lun Chao\*, Fei Sha, and Kristen Grauman. 
<br>
In Proceedings of the European Conference on Computer Vision (ECCV), 2016, Amsterdam, The Netherlands. (*Equal contribution)  \[[pdf](http://www-scf.usc.edu/~zhan355/ke_eccv2016.pdf)\] 

**[Supplementary Material](http://www-scf.usc.edu/~zhan355/ke_eccv2016_supp.pdf)**

If you find the codes or other related resources from this repository useful, please cite the following paper:

```
@inproceedings{zhang2016video,
  title={Video summarization with long short-term memory},
  author={Zhang, Ke and Chao, Wei-Lun and Sha, Fei and Grauman, Kristen},
  booktitle={ECCV},
  year={2016},
  organization={Springer}
}
```


## Environment

- MAC OS X or Linux
- NVIDIA GPU with compute capability 3.5+
- Python 2.7+
- Theano 0.7+
- Matlab

## Data

Download the [data](https://www.dropbox.com/s/ynl4jsa2mxohs16/data.zip?dl=0) and unzip to *./data/*

Note that we down-sampled the original video by 2fps. 
1) file name: in the format 'Data_$Dataset$_google_p5.h5', e.g. Data_SumMe_google_p5.h5, means the frame level feature of SumMe dataset. 
2) the index of videos are stored as ‘idx’ in the file, in most cases it’s from 1 to n, where n is the number of videos in the dataset (except for Youtube dataset).
3) feature & ground-truth: the feature is indexed as ‘fea_i’ , the importance is indexed as ‘gt_1_i’ (real number, from the original dataset), and the keyframe we used is indexed as ‘gt_2_i’  (binary value transferred from the original dataset) for the i-th video in the dataset.

Original videos and annotations for each dataset are also available from the the authors' project page
* TVSum dataset: https://github.com/yalesong/tvsum
* SumMe dataset: https://people.ee.ethz.ch/~gyglim/vsum/#benchmark
* OVP and YouTube datasets: https://sites.google.com/site/vsummsite/

## Codes

### dppLSTM for video summarization
We have enclosed pre-trained models in the *./model* directory
download the model and run the following commands:

Download the pre-trained models and unzip it to *./models* and run the following commands:
```
cd ./codes
THEANO_FLAGS=device=gpu0,floatX=float32 python dppLSTM_main.py 
```

This will automatically run summarization on the video data using pre-trained model, and save the results in *./res_LSTM/* as **dppLSTM_$DATASET$_2_inference.h5**

If you want to train the model on your own data, just uncomment Line 85 in *dppLSTM_main.py*
```
train(model_idx = model_idx, train_set = train_set, val_set = val_set, model_saved = model_file)
```
### Evaluation

For both SumMe and TVSum datasets, you can find the code for evaluation provided by the author:
* TVSum: https://github.com/yalesong/tvsum
* SumMe: https://people.ee.ethz.ch/~gyglim/vsum/#benchmark

We also provided the evaluation code with wrappers that help adapt to the datasets above

To run evaluation on the predicted summarization, start the matlab and run the following commands: 
```
cd ./codes
dppLSTM_eval('../data/', '$DATASET$', '/dppLSTM_$DATASET$_2_inference.h5')
```
---
### Notes
Important [issues thread](https://github.com/kezhang-cs/Video-Summarization-with-LSTM/issues/1) from original repo: 
- Folder res_LSTM should be created in /codes.
- Meaning of output fields: 
..>pred and predk are the corresponding prediction of y_i for each frame and \phi_i for DPP.

Another comment from original repo on how they extracted features from video
>We used caffe to extract feature for each frame, the code is available from here:
>https://github.com/BVLC/caffe/tree/master/models/bvlc_googlenet


>For all models, we set the size of each
>hidden layer of MLPs, the number of hidden units of each unidirectional LSTM,
>and the output dimension of the MLP fS(·) all to be 256.

[Paper](https://hal.inria.fr/hal-01022967/PDF/video_summarization.pdf) about Kernel Temporal Segmentation used for selecting frames based on inferred ratings and corresponding [code](http://pascal.inrialpes.fr/data2/potapov/med_summaries/kts_ver1.1.tar.gz)
