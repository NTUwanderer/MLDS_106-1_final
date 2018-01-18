# MLDS_106-1_final
MLDS final project, 106-1, HTC hand detection

#### Environment Description
Recommended System: Ubuntu 16.04 <br />
Required Data: *DeepQ-Synth-Hand-01  DeepQ-Synth-Hand-02  DeepQ-Vivepaper* in the same path (described as *path directory* below).
Required Packages & Version:
- keras 2.0.8
- tensorflow 1.4.0
- tqdm 4.19.4
- imgaug 0.2.5
- progressbar2 3.34.3
- opencv-pytnon 3.3.0.10
- numpy 1.13.3

#### Install Package
```bash
$ pip3 install -r requirements.txt
```

#### Download Trained Model
Our trained hand detection model is available at gitlab. Run the download script to download the model:
```bash
$ bash download.sh
```

#### Test Model
Make sure you have the model in your directory (either downloaded or trained by yourself). <br />
Make sure you have prepared the dataset and setup the environment. <br />
The default *path to model* is *full_yolo_hand_small.h5* 
```bash
$ python3 upload.py –w <path_to_model>
```

#### Train Model (Yolo2)
Our default configuration of training process is written in config.json. You do not have to change the configuration to run a training task. <br />
The default *path to model* is *full_yolo_hand_small.h5*
```bash
$ python3 train.py -w <path_to_model> -i <data directory>
```
Note: If you train the model from scratch, the result might be slightly different from our trained model.

