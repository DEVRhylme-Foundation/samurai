<div align="center">
<img align="left" width="100" height="100" src="https://github.com/user-attachments/assets/1834fc25-42ef-4237-9feb-53a01c137e83" alt="">

# SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

</div>






SAMURAI is a cutting-edge visual tracking model that adapts the Segment Anything Model (SAM) for zero-shot tracking tasks. It leverages motion-aware memory to enhance tracking performance without requiring additional training. SAMURAI utilizes the pre-trained weights from SAM 2.1 and integrates a Kalman filter for accurate object tracking across frames. This approach is particularly effective for various visual object tracking benchmarks, including LaSOT, GOT-10k, and OTB-2015.

## 🔑Key Features
- **Zero-Shot Tracking:** SAMURAI does not require additional training and directly uses the pre-trained SAM 2.1 weights.
- **Motion-Aware Memory:**  Incorporates a Kalman filter to estimate and predict object motion, improving tracking robustness
- **State-of-the-Art Performance** Achieves top results on multiple VOT benchmarks, including LaSOT, GOT-10k, NFS, and OTB. 

## 🎥DEMO VIDEO

https://github.com/user-attachments/assets/9d368ca7-2e9b-4fed-9da0-d2efbf620d88



## 📘Usage Guide


### 🛠️SAMURAI Installation 

#### Step 1: Install Python
First, ensure you have Python installed on your system. SAMURAI requires Python 3.10 or higher. You can download Python from the official website. [here](https://python.org) 

#### Step 2: Set Up a Virtual Environment
It's a good practice to use a virtual environment to manage dependencies. Open your terminal or command prompt and run the following commands:

Install virtualenv if you don't have it
``` Bash
pip install virtualenv
```

Create a virtual environment named 'samurai_env'
``` Bash
virtualenv samurai_env
```

**Activate the virtual environment**

On Windows
``` Bash
samurai_env\Scripts\activate
```

On macOS/Linux
```Bash
source samurai_env/bin/activate
````
#### Step 3: Install SAM 2
Clone the SAM 2 repository:

``` bash
git clone https://github.com/DEVRhylme-Foundation/samurai.git
cd sam2
```
Install SAM 2
``` bash
pip install -e .
pip install -e ".[notebooks]"
````

#### Step 4: Install Additional Requirements
Install the additional dependencies required by SAMURAI:
``` bash
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
```
#### Step 5: Download SAM 2.1 Checkpoints
Download the necessary checkpoints for SAM 2.1:
``` bash
cd checkpoints
./download_ckpts.sh
cd ..
```
#### Step 6: Prepare Your Data
Organize your data in the following format:
``` bash
data/LaSOT
├── airplane/
│   ├── airplane-1/
│   │   ├── full_occlusion.txt
│   │   ├── groundtruth.txt
│   │   ├── img
│   │   ├── nlp.txt
│   │   └── out_of_view.txt
│   ├── airplane-2/
│   ├── airplane-3/
│   ├── ...
├── basketball
├── bear
├── bicycle
...
├── training_set.txt
└── testing_set.txt
```
#### Step 7: Run Main Inference
Execute the main inference script to start tracking:

```bash
python scripts/main_inference.py 
```
#### Step 8: Demo on Custom Video
To run the demo with your custom video or frame directory, use the following commands:

##### Input is Video File:
``` bash
python scripts/demo.py --video_path <your_video.mp4> --txt_path <path_to_first_frame_bbox.txt>
```
##### Input is Frame Folder:

Only JPG images are supported
``` bash
python scripts/demo.py --video_path <your_frame_directory> --txt_path <path_to_first_frame_bbox.txt>
```


## FAQs
**Question 1:** Does SAMURAI need training? [issue 34](https://github.com/yangchris11/samurai/issues/34)

**Answer 1:** Unlike real-life samurai, the proposed samurai do not require additional training. It is a zero-shot method, we directly use the weights from SAM 2.1 to conduct VOT experiments. The Kalman filter is used to estimate the current and future state (bounding box location and scale in our case) of a moving object based on measurements over time, it is a common approach that had been adopted in the field of tracking for a long time, which does not require any training. Please refer to code for more detail.

**Question 2:** Does SAMURAI support streaming input (e.g. webcam)?

**Answer 2:** Not yet. The existing code doesn't support live/streaming video as we inherit most of the codebase from the amazing SAM 2. Some discussion that you might be interested in: facebookresearch/sam2#90, facebookresearch/sam2#388 (comment).

**Question 3:** How to use SAMURAI in longer video?

**Answer 3:** See the discussion from sam2 https://github.com/facebookresearch/sam2/issues/264.

**Question 4:** How do you run the evaluation on the VOT benchmarks?

**Answer 4:** For LaSOT, LaSOT-ext, OTB, NFS please refer to the [issue 74](https://github.com/yangchris11/samurai/issues/74) for more details. For GOT-10k-test and TrackingNet, please refer to the official portal for submission.

## Acknowledgment

SAMURAI is built on top of [SAM 2](https://github.com/facebookresearch/sam2?tab=readme-ov-file) by Meta FAIR.

The VOT evaluation code is modifed from [VOT Toolkit](https://github.com/votchallenge/toolkit) by Luka Čehovin Zajc.

## Citation

Please consider citing our paper and the wonderful `SAM 2` if you found our work interesting and useful.
```
@article{ravi2024sam2,
  title={SAM 2: Segment Anything in Images and Videos},
  author={Ravi, Nikhila and Gabeur, Valentin and Hu, Yuan-Ting and Hu, Ronghang and Ryali, Chaitanya and Ma, Tengyu and Khedr, Haitham and R{\"a}dle, Roman and Rolland, Chloe and Gustafson, Laura and Mintun, Eric and Pan, Junting and Alwala, Kalyan Vasudev and Carion, Nicolas and Wu, Chao-Yuan and Girshick, Ross and Doll{\'a}r, Piotr and Feichtenhofer, Christoph},
  journal={arXiv preprint arXiv:2408.00714},
  url={https://arxiv.org/abs/2408.00714},
  year={2024}
}

@misc{yang2024samurai,
  title={SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory}, 
  author={Cheng-Yen Yang and Hsiang-Wei Huang and Wenhao Chai and Zhongyu Jiang and Jenq-Neng Hwang},
  year={2024},
  eprint={2411.11922},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2411.11922}, 
}
```
