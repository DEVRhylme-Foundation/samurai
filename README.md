<div align="center">
<img align="left" width="100" height="100" src="https://github.com/user-attachments/assets/1834fc25-42ef-4237-9feb-53a01c137e83" alt="">

# SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

[Cheng-Yen Yang](https://yangchris11.github.io), [Hsiang-Wei Huang](https://hsiangwei0903.github.io/), [Wenhao Chai](https://rese1f.github.io/), [Zhongyu Jiang](https://zhyjiang.github.io/#/), [Jenq-Neng Hwang](https://people.ece.uw.edu/hwang/)

[Information Processing Lab, University of Washington](https://ipl-uw.github.io/) 
</div>


[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-lasot-ext)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot-ext?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-got-10k)](https://paperswithcode.com/sota/visual-object-tracking-on-got-10k?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-needforspeed)](https://paperswithcode.com/sota/visual-object-tracking-on-needforspeed?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-lasot)](https://paperswithcode.com/sota/visual-object-tracking-on-lasot?p=samurai-adapting-segment-anything-model-for-1)
[![PWC](https://img.shields.io/endpoint.svg?url=https://paperswithcode.com/badge/samurai-adapting-segment-anything-model-for-1/visual-object-tracking-on-otb-2015)](https://paperswithcode.com/sota/visual-object-tracking-on-otb-2015?p=samurai-adapting-segment-anything-model-for-1)

[[Arxiv]](https://arxiv.org/abs/2411.11922) [[Project Page]](https://yangchris11.github.io/samurai/) [[Raw Results]](https://drive.google.com/drive/folders/1ssiDmsC7mw5AiItYQG4poiR1JgRq305y?usp=sharing) 

This repository is the official implementation of SAMURAI: Adapting Segment Anything Model for Zero-Shot Visual Tracking with Motion-Aware Memory

https://github.com/user-attachments/assets/9d368ca7-2e9b-4fed-9da0-d2efbf620d88

All rights are reserved to the copyright owners (TM & © Universal (2019)). This clip is not intended for commercial use and is solely for academic demonstration in a research paper. Original source can be found [here](https://www.youtube.com/watch?v=cwUzUzpG8aM&t=4s).

## Getting Started

#### SAMURAI Installation 

SAM 2 needs to be installed first before use. The code requires `python>=3.10`, as well as `torch>=2.3.1` and `torchvision>=0.18.1`. Please follow the instructions [here](https://github.com/facebookresearch/sam2?tab=readme-ov-file) to install both PyTorch and TorchVision dependencies. You can install **the SAMURAI version** of SAM 2 on a GPU machine using:
```
cd sam2
pip install -e .
pip install -e ".[notebooks]"
```

Please see [INSTALL.md](https://github.com/facebookresearch/sam2/blob/main/INSTALL.md) from the original SAM 2 repository for FAQs on potential issues and solutions.

Install other requirements:

```
pip install matplotlib==3.7 tikzplotlib jpeg4py opencv-python lmdb pandas scipy loguru
```

#### SAM 2.1 Checkpoint Download

```
cd checkpoints && \
./download_ckpts.sh && \
cd ..
```

#### Data Preparation

Please prepare the data in the following format:
```
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

#### Main Inference
```
python scripts/main_inference.py 
```

## Demo on Custom Video

To run the demo with your custom video or frame directory, use the following examples:

**Note:** The `.txt` file contains a single line with the bounding box of the first frame in `x,y,w,h` format while the SAM 2 takes `x1,y1,x2,y2` format as bbox input.

### Input is Video File

```
python scripts/demo.py --video_path <your_video.mp4> --txt_path <path_to_first_frame_bbox.txt>
```

### Input is Frame Folder
```
# Only JPG images are supported
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

<details>
   <summary>
      <h2>Repository Structure</h2>
   </summary>

   The Repository structure of the project is as follows:
<!-- START_STRUCTURE -->
```
├── LICENSE
├── README.md
├── assets
│   └── samurai_demo.mp4
├── data
├── lib
│   ├── test
│   │   ├── __init__.py
│   │   ├── analysis
│   │   │   ├── __init__.py
│   │   │   ├── extract_results.py
│   │   │   └── plot_results.py
│   │   ├── evaluation
│   │   │   ├── __init__.py
│   │   │   ├── data.py
│   │   │   ├── datasets.py
│   │   │   ├── environment.py
│   │   │   ├── got10kdataset.py
│   │   │   ├── itbdataset.py
│   │   │   ├── lasot_lmdbdataset.py
│   │   │   ├── lasotdataset.py
│   │   │   ├── lasotextensionsubsetdataset.py
│   │   │   ├── local.py
│   │   │   ├── nfsdataset.py
│   │   │   ├── otbdataset.py
│   │   │   ├── running.py
│   │   │   ├── tc128cedataset.py
│   │   │   ├── tc128dataset.py
│   │   │   ├── tnl2kdataset.py
│   │   │   ├── tracker.py
│   │   │   ├── trackingnetdataset.py
│   │   │   ├── uavdataset.py
│   │   │   └── votdataset.py
│   │   ├── parameter
│   │   │   ├── __init__.py
│   │   │   ├── artrack.py
│   │   │   └── artrack_seq.py
│   │   ├── tracker
│   │   │   ├── __init__.py
│   │   │   ├── artrack.py
│   │   │   ├── artrack_seq.py
│   │   │   ├── basetracker.py
│   │   │   ├── data_utils.py
│   │   │   └── vis_utils.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── _init_paths.py
│   │       ├── hann.py
│   │       ├── load_text.py
│   │       ├── params.py
│   │       ├── transform_got10k.py
│   │       └── transform_trackingnet.py
│   ├── train
│   │   ├── __init__.py
│   │   ├── _init_paths.py
│   │   ├── actors
│   │   │   ├── __init__.py
│   │   │   ├── artrack.py
│   │   │   ├── artrack_seq.py
│   │   │   └── base_actor.py
│   │   ├── admin
│   │   │   ├── __init__.py
│   │   │   ├── environment.py
│   │   │   ├── local.py
│   │   │   ├── multigpu.py
│   │   │   ├── settings.py
│   │   │   ├── stats.py
│   │   │   └── tensorboard.py
│   │   ├── base_functions.py
│   │   ├── data
│   │   │   ├── __init__.py
│   │   │   ├── bounding_box_utils.py
│   │   │   ├── image_loader.py
│   │   │   ├── loader.py
│   │   │   ├── processing.py
│   │   │   ├── processing_utils.py
│   │   │   ├── sampler.py
│   │   │   ├── sequence_sampler.py
│   │   │   ├── transforms.py
│   │   │   └── wandb_logger.py
│   │   ├── data_specs
│   │   │   ├── README.md
│   │   │   ├── got10k_train_full_split.txt
│   │   │   ├── got10k_train_split.txt
│   │   │   ├── got10k_val_split.txt
│   │   │   ├── got10k_vot_exclude.txt
│   │   │   ├── got10k_vot_train_split.txt
│   │   │   ├── got10k_vot_val_split.txt
│   │   │   ├── lasot_train_split.txt
│   │   │   └── trackingnet_classmap.txt
│   │   ├── dataset
│   │   │   ├── COCO_tool.py
│   │   │   ├── __init__.py
│   │   │   ├── base_image_dataset.py
│   │   │   ├── base_video_dataset.py
│   │   │   ├── coco.py
│   │   │   ├── coco_seq.py
│   │   │   ├── coco_seq_lmdb.py
│   │   │   ├── got10k.py
│   │   │   ├── got10k_lmdb.py
│   │   │   ├── imagenetvid.py
│   │   │   ├── imagenetvid_lmdb.py
│   │   │   ├── lasot.py
│   │   │   ├── lasot_lmdb.py
│   │   │   ├── tracking_net.py
│   │   │   └── tracking_net_lmdb.py
│   │   ├── run_training.py
│   │   ├── train_script.py
│   │   ├── train_script_distill.py
│   │   └── trainers
│   │       ├── __init__.py
│   │       ├── base_trainer.py
│   │       ├── ltr_seq_trainer.py
│   │       └── ltr_trainer.py
│   └── utils
│       ├── __init__.py
│       ├── box_ops.py
│       ├── ce_utils.py
│       ├── focal_loss.py
│       ├── heapmap_utils.py
│       ├── lmdb_utils.py
│       ├── merge.py
│       ├── misc.py
│       ├── tensor.py
│       └── variable_hook.py
├── sam2
│   ├── CODE_OF_CONDUCT.md
│   ├── CONTRIBUTING.md
│   ├── INSTALL.md
│   ├── LICENSE
│   ├── LICENSE_cctorch
│   ├── MANIFEST.in
│   ├── README.md
│   ├── assets
│   │   ├── model_diagram.png
│   │   └── sa_v_dataset.jpg
│   ├── backend.Dockerfile
│   ├── checkpoints
│   │   └── download_ckpts.sh
│   ├── demo
│   │   ├── README.md
│   │   ├── backend
│   │   │   └── server
│   │   │       ├── app.py
│   │   │       ├── app_conf.py
│   │   │       ├── data
│   │   │       │   ├── data_types.py
│   │   │       │   ├── loader.py
│   │   │       │   ├── resolver.py
│   │   │       │   ├── schema.py
│   │   │       │   ├── store.py
│   │   │       │   └── transcoder.py
│   │   │       └── inference
│   │   │           ├── data_types.py
│   │   │           ├── multipart.py
│   │   │           └── predictor.py
│   │   ├── data
│   │   │   └── gallery
│   │   │       ├── 01_dog.mp4
│   │   │       ├── 02_cups.mp4
│   │   │       ├── 03_blocks.mp4
│   │   │       ├── 04_coffee.mp4
│   │   │       └── 05_default_juggle.mp4
│   │   └── frontend
│   │       ├── frontend.Dockerfile
│   │       ├── index.html
│   │       ├── package.json
│   │       ├── postcss.config.js
│   │       ├── public
│   │       │   └── fonts
│   │       │       └── Inter-VariableFont_opsz,wght.ttf
│   │       ├── schema.graphql
│   │       ├── schemas
│   │       │   ├── inference-api-schema.graphql
│   │       │   ├── merge-schemas.ts
│   │       │   └── video-api-schema.graphql
│   │       ├── src
│   │       │   ├── App.tsx
│   │       │   ├── assets
│   │       │   │   ├── icons
│   │       │   │   │   ├── angery.png
│   │       │   │   │   ├── heart.png
│   │       │   │   │   └── whistle.png
│   │       │   │   ├── scss
│   │       │   │   │   └── App.scss
│   │       │   │   └── videos
│   │       │   │       ├── sam2_720px_dark.mp4
│   │       │   │       └── sam2_video_poster.png
│   │       │   ├── common
│   │       │   │   ├── codecs
│   │       │   │   │   ├── VideoDecoder.ts
│   │       │   │   │   ├── VideoEncoder.ts
│   │       │   │   │   └── WebCodecUtils.ts
│   │       │   │   ├── components
│   │       │   │   │   ├── MobileFirstClickBanner.tsx
│   │       │   │   │   ├── Tooltip.tsx
│   │       │   │   │   ├── annotations
│   │       │   │   │   │   ├── AddObjectButton.tsx
│   │       │   │   │   │   ├── ClearAllPointsInVideoButton.tsx
│   │       │   │   │   │   ├── CloseSessionButton.tsx
│   │       │   │   │   │   ├── FirstClickView.tsx
│   │       │   │   │   │   ├── LimitNotice.tsx
│   │       │   │   │   │   ├── MobileObjectsList.tsx
│   │       │   │   │   │   ├── MobileObjectsToolbar.tsx
│   │       │   │   │   │   ├── MobileObjectsToolbarHeader.tsx
│   │       │   │   │   │   ├── ObjectActions.tsx
│   │       │   │   │   │   ├── ObjectPlaceholder.tsx
│   │       │   │   │   │   ├── ObjectThumbnail.tsx
│   │       │   │   │   │   ├── ObjectUtils.ts
│   │       │   │   │   │   ├── ObjectsToolbar.tsx
│   │       │   │   │   │   ├── ObjectsToolbarBottomActions.tsx
│   │       │   │   │   │   ├── ObjectsToolbarHeader.tsx
│   │       │   │   │   │   ├── PointsToggle.tsx
│   │       │   │   │   │   ├── PrimaryCTAButton.tsx
│   │       │   │   │   │   ├── ToolbarObject.tsx
│   │       │   │   │   │   ├── ToolbarObjectContainer.tsx
│   │       │   │   │   │   ├── TrackletSwimlane.tsx
│   │       │   │   │   │   ├── TrackletsAnnotation.tsx
│   │       │   │   │   │   └── useTracklets.ts
│   │       │   │   │   ├── button
│   │       │   │   │   │   ├── GradientBorder.tsx
│   │       │   │   │   │   ├── PlaybackButton.tsx
│   │       │   │   │   │   ├── PrimaryCTAButton.tsx
│   │       │   │   │   │   ├── ResponsiveButton.tsx
│   │       │   │   │   │   └── TrackAndPlayButton.tsx
│   │       │   │   │   ├── code
│   │       │   │   │   │   └── InitializeLocalMonaco.ts
│   │       │   │   │   ├── effects
│   │       │   │   │   │   ├── BackgroundEffects.tsx
│   │       │   │   │   │   ├── EffectVariantBadge.tsx
│   │       │   │   │   │   ├── EffectsCarousel.tsx
│   │       │   │   │   │   ├── EffectsCarouselShadow.tsx
│   │       │   │   │   │   ├── EffectsToolbar.tsx
│   │       │   │   │   │   ├── EffectsToolbarBottomActions.tsx
│   │       │   │   │   │   ├── EffectsToolbarHeader.tsx
│   │       │   │   │   │   ├── EffectsUtils.ts
│   │       │   │   │   │   ├── HighlightEffects.tsx
│   │       │   │   │   │   ├── MobileEffectsToolbar.tsx
│   │       │   │   │   │   └── MoreFunEffects.tsx
│   │       │   │   │   ├── gallery
│   │       │   │   │   │   ├── ChangeVideoModal.tsx
│   │       │   │   │   │   ├── DefaultVideoGalleryModalTrigger.tsx
│   │       │   │   │   │   ├── DemoVideoGallery.tsx
│   │       │   │   │   │   ├── DemoVideoGalleryModal.tsx
│   │       │   │   │   │   ├── VideoGalleryUploadPhoto.tsx
│   │       │   │   │   │   ├── VideoPhoto.tsx
│   │       │   │   │   │   ├── __generated__
│   │       │   │   │   │   │   ├── DemoVideoGalleryModalQuery.graphql.ts
│   │       │   │   │   │   │   ├── DemoVideoGalleryQuery.graphql.ts
│   │       │   │   │   │   │   └── useUploadVideoMutation.graphql.ts
│   │       │   │   │   │   └── useUploadVideo.ts
│   │       │   │   │   ├── icons
│   │       │   │   │   │   └── GitHubIcon.tsx
│   │       │   │   │   ├── options
│   │       │   │   │   │   ├── DownloadOption.tsx
│   │       │   │   │   │   ├── GalleryOption.tsx
│   │       │   │   │   │   ├── MoreOptionsToolbar.tsx
│   │       │   │   │   │   ├── MoreOptionsToolbarBottomActions.tsx
│   │       │   │   │   │   ├── OptionButton.tsx
│   │       │   │   │   │   ├── ShareSection.tsx
│   │       │   │   │   │   ├── ShareUtils.ts
│   │       │   │   │   │   ├── TryAnotherVideoSection.tsx
│   │       │   │   │   │   ├── UploadOption.tsx
│   │       │   │   │   │   ├── __generated__
│   │       │   │   │   │   │   └── GetLinkOptionShareVideoMutation.graphql.ts
│   │       │   │   │   │   └── useDownloadVideo.ts
│   │       │   │   │   ├── session
│   │       │   │   │   │   ├── RestartSessionButton.tsx
│   │       │   │   │   │   ├── __generated__
│   │       │   │   │   │   │   └── useCloseSessionBeforeUnloadMutation.graphql.ts
│   │       │   │   │   │   ├── useCloseSessionBeforeUnload.ts
│   │       │   │   │   │   └── useRestartSession.ts
│   │       │   │   │   ├── snackbar
│   │       │   │   │   │   ├── DemoMessagesSnackbarUtils.ts
│   │       │   │   │   │   ├── MessagesSnackbar.tsx
│   │       │   │   │   │   ├── snackbarAtoms.ts
│   │       │   │   │   │   ├── useDemoMessagesSnackbar.ts
│   │       │   │   │   │   ├── useExpireMessage.ts
│   │       │   │   │   │   └── useMessagesSnackbar.ts
│   │       │   │   │   ├── toolbar
│   │       │   │   │   │   ├── DesktopToolbar.tsx
│   │       │   │   │   │   ├── MobileToolbar.tsx
│   │       │   │   │   │   ├── Toolbar.tsx
│   │       │   │   │   │   ├── ToolbarActionIcon.tsx
│   │       │   │   │   │   ├── ToolbarBottomActionsWrapper.tsx
│   │       │   │   │   │   ├── ToolbarConfig.tsx
│   │       │   │   │   │   ├── ToolbarHeaderWrapper.tsx
│   │       │   │   │   │   ├── ToolbarProgressChip.tsx
│   │       │   │   │   │   ├── ToolbarSection.tsx
│   │       │   │   │   │   ├── useListenToStreamingState.ts
│   │       │   │   │   │   └── useToolbarTabs.ts
│   │       │   │   │   ├── useFunctionThrottle.tsx
│   │       │   │   │   └── video
│   │       │   │   │       ├── ChangeVideoModal.tsx
│   │       │   │   │       ├── EventEmitter.ts
│   │       │   │   │       ├── Video.tsx
│   │       │   │   │       ├── VideoFilmstripWithPlayback.tsx
│   │       │   │   │       ├── VideoLoadingOverlay.tsx
│   │       │   │   │       ├── VideoWorker.ts
│   │       │   │   │       ├── VideoWorkerBridge.ts
│   │       │   │   │       ├── VideoWorkerContext.ts
│   │       │   │   │       ├── VideoWorkerTypes.ts
│   │       │   │   │       ├── editor
│   │       │   │   │       │   ├── DemoVideoEditor.tsx
│   │       │   │   │       │   ├── ImageUtils.ts
│   │       │   │   │       │   ├── VideoEditor.tsx
│   │       │   │   │       │   ├── VideoEditorUtils.ts
│   │       │   │   │       │   ├── atoms.ts
│   │       │   │   │       │   ├── useResetEditor.ts
│   │       │   │   │       │   ├── useVideo.ts
│   │       │   │   │       │   └── useVideoEffect.ts
│   │       │   │   │       ├── effects
│   │       │   │   │       │   ├── ArrowGLEffect.ts
│   │       │   │   │       │   ├── BackgroundBlurEffect.ts
│   │       │   │   │       │   ├── BackgroundTextEffect.ts
│   │       │   │   │       │   ├── BaseGLEffect.ts
│   │       │   │   │       │   ├── BurstGLEffect.ts
│   │       │   │   │       │   ├── CutoutGLEffect.ts
│   │       │   │   │       │   ├── DesaturateEffect.ts
│   │       │   │   │       │   ├── Effect.ts
│   │       │   │   │       │   ├── EffectUtils.ts
│   │       │   │   │       │   ├── Effects.ts
│   │       │   │   │       │   ├── EraseBackgroundEffect.ts
│   │       │   │   │       │   ├── EraseForegroundEffect.ts
│   │       │   │   │       │   ├── EraseForegroundGLEffect.ts
│   │       │   │   │       │   ├── GradientEffect.ts
│   │       │   │   │       │   ├── NoisyMaskEffect.ts
│   │       │   │   │       │   ├── OriginalEffect.ts
│   │       │   │   │       │   ├── OverlayEffect.ts
│   │       │   │   │       │   ├── PixelateEffect.ts
│   │       │   │   │       │   ├── PixelateMaskGLEffect.ts
│   │       │   │   │       │   ├── ReplaceGLEffect.ts
│   │       │   │   │       │   ├── ScopeGLEffect.ts
│   │       │   │   │       │   ├── SobelEffect.ts
│   │       │   │   │       │   ├── VibrantMaskEffect.ts
│   │       │   │   │       │   └── shaders
│   │       │   │   │       │       ├── Arrow.frag
│   │       │   │   │       │       ├── BackgroundBlur.frag
│   │       │   │   │       │       ├── Burst.frag
│   │       │   │   │       │       ├── Cutout.frag
│   │       │   │   │       │       ├── DefaultVert.vert
│   │       │   │   │       │       ├── EraseForeground.frag
│   │       │   │   │       │       ├── Gradient.frag
│   │       │   │   │       │       ├── NoisyMask.frag
│   │       │   │   │       │       ├── Overlay.frag
│   │       │   │   │       │       ├── Overlay.vert
│   │       │   │   │       │       ├── Pixelate.frag
│   │       │   │   │       │       ├── PixelateMask.frag
│   │       │   │   │       │       ├── Replace.frag
│   │       │   │   │       │       ├── Scope.frag
│   │       │   │   │       │       ├── Sobel.frag
│   │       │   │   │       │       └── VibrantMask.frag
│   │       │   │   │       ├── filmstrip
│   │       │   │   │       │   ├── FilmstripUtil.tsx
│   │       │   │   │       │   ├── SelectedFrameHelper.ts
│   │       │   │   │       │   ├── VideoFilmstrip.tsx
│   │       │   │   │       │   ├── atoms.ts
│   │       │   │   │       │   ├── useDisableScrolling.ts
│   │       │   │   │       │   └── useSelectedFrameHelper.ts
│   │       │   │   │       ├── layers
│   │       │   │   │       │   ├── InteractionLayer.tsx
│   │       │   │   │       │   └── PointsLayer.tsx
│   │       │   │   │       ├── useInputVideo.ts
│   │       │   │   │       └── useVideoWorker.ts
│   │       │   │   ├── error
│   │       │   │   │   ├── ErrorFallback.tsx
│   │       │   │   │   ├── ErrorReport.tsx
│   │       │   │   │   ├── ErrorSerializationUtils.ts
│   │       │   │   │   ├── ErrorUtils.ts
│   │       │   │   │   ├── errorReportAtom.ts
│   │       │   │   │   └── useReportError.tsx
│   │       │   │   ├── loading
│   │       │   │   │   ├── LoadingMessage.tsx
│   │       │   │   │   ├── LoadingStateScreen.tsx
│   │       │   │   │   ├── StaticVideoPlayer.tsx
│   │       │   │   │   └── UploadLoadingScreen.tsx
│   │       │   │   ├── logger
│   │       │   │   │   ├── DemoLogger.ts
│   │       │   │   │   ├── LogEnvironment.ts
│   │       │   │   │   └── Logger.ts
│   │       │   │   ├── screen
│   │       │   │   │   └── useScreenSize.tsx
│   │       │   │   ├── tracker
│   │       │   │   │   ├── SAM2Model.ts
│   │       │   │   │   ├── Tracker.ts
│   │       │   │   │   ├── TrackerTypes.ts
│   │       │   │   │   ├── Trackers.ts
│   │       │   │   │   └── __generated__
│   │       │   │   │       ├── SAM2ModelAddNewPointsMutation.graphql.ts
│   │       │   │   │       ├── SAM2ModelCancelPropagateInVideoMutation.graphql.ts
│   │       │   │   │       ├── SAM2ModelClearPointsInFrameMutation.graphql.ts
│   │       │   │   │       ├── SAM2ModelClearPointsInVideoMutation.graphql.ts
│   │       │   │   │       ├── SAM2ModelCloseSessionMutation.graphql.ts
│   │       │   │   │       ├── SAM2ModelRemoveObjectMutation.graphql.ts
│   │       │   │   │       └── SAM2ModelStartSessionMutation.graphql.ts
│   │       │   │   └── utils
│   │       │   │       ├── FileUtils.ts
│   │       │   │       ├── ImageUtils.ts
│   │       │   │       ├── MaskUtils.ts
│   │       │   │       ├── MultipartStream.ts
│   │       │   │       ├── ShaderUtils.ts
│   │       │   │       ├── emptyFunction.ts
│   │       │   │       └── uuid.ts
│   │       │   ├── debug
│   │       │   │   └── stats
│   │       │   │       ├── Stats.ts
│   │       │   │       └── StatsView.tsx
│   │       │   ├── demo
│   │       │   │   ├── DemoConfig.tsx
│   │       │   │   ├── DemoErrorFallback.tsx
│   │       │   │   ├── DemoSuspenseFallback.tsx
│   │       │   │   ├── SAM2DemoApp.tsx
│   │       │   │   └── atoms.ts
│   │       │   ├── graphql
│   │       │   │   ├── RelayEnvironment.ts
│   │       │   │   ├── RelayEnvironmentProvider.tsx
│   │       │   │   ├── errors
│   │       │   │   │   ├── CreateFilmstripError.ts
│   │       │   │   │   ├── DrawFrameError.ts
│   │       │   │   │   └── WebGLContextError.ts
│   │       │   │   └── fetchGraphQL.ts
│   │       │   ├── jscocotools
│   │       │   │   └── mask.ts
│   │       │   ├── layouts
│   │       │   │   ├── DemoPageLayout.tsx
│   │       │   │   └── RootLayout.tsx
│   │       │   ├── main.tsx
│   │       │   ├── routes
│   │       │   │   ├── DemoPage.tsx
│   │       │   │   ├── DemoPageWrapper.tsx
│   │       │   │   ├── PageNotFoundPage.tsx
│   │       │   │   └── __generated__
│   │       │   │       └── DemoPageQuery.graphql.ts
│   │       │   ├── settings
│   │       │   │   ├── ApprovableInput.tsx
│   │       │   │   ├── SAM2Settings.tsx
│   │       │   │   ├── SettingsContextProvider.tsx
│   │       │   │   ├── SettingsModal.tsx
│   │       │   │   ├── SettingsReducer.ts
│   │       │   │   └── useSettingsContext.tsx
│   │       │   ├── theme
│   │       │   │   ├── colors.ts
│   │       │   │   ├── gradientStyle.ts
│   │       │   │   └── tokens.stylex.ts
│   │       │   ├── types
│   │       │   │   └── mp4box
│   │       │   │       └── index.d.ts
│   │       │   └── vite-env.d.ts
│   │       ├── tailwind.config.js
│   │       ├── tsconfig.json
│   │       ├── tsconfig.node.json
│   │       ├── vite.config.ts
│   │       └── yarn.lock
│   ├── docker-compose.yaml
│   ├── pyproject.toml
│   ├── sam2
│   │   ├── __init__.py
│   │   ├── automatic_mask_generator.py
│   │   ├── build_sam.py
│   │   ├── configs
│   │   │   ├── sam2
│   │   │   │   ├── sam2_hiera_b+.yaml
│   │   │   │   ├── sam2_hiera_l.yaml
│   │   │   │   ├── sam2_hiera_s.yaml
│   │   │   │   └── sam2_hiera_t.yaml
│   │   │   ├── sam2.1
│   │   │   │   ├── sam2.1_hiera_b+.yaml
│   │   │   │   ├── sam2.1_hiera_l.yaml
│   │   │   │   ├── sam2.1_hiera_s.yaml
│   │   │   │   └── sam2.1_hiera_t.yaml
│   │   │   ├── sam2.1_training
│   │   │   │   └── sam2.1_hiera_b+_MOSE_finetune.yaml
│   │   │   └── samurai
│   │   │       ├── sam2.1_hiera_b+.yaml
│   │   │       ├── sam2.1_hiera_l.yaml
│   │   │       ├── sam2.1_hiera_s.yaml
│   │   │       └── sam2.1_hiera_t.yaml
│   │   ├── csrc
│   │   │   └── connected_components.cu
│   │   ├── modeling
│   │   │   ├── __init__.py
│   │   │   ├── backbones
│   │   │   │   ├── __init__.py
│   │   │   │   ├── hieradet.py
│   │   │   │   ├── image_encoder.py
│   │   │   │   └── utils.py
│   │   │   ├── memory_attention.py
│   │   │   ├── memory_encoder.py
│   │   │   ├── position_encoding.py
│   │   │   ├── sam
│   │   │   │   ├── __init__.py
│   │   │   │   ├── mask_decoder.py
│   │   │   │   ├── prompt_encoder.py
│   │   │   │   └── transformer.py
│   │   │   ├── sam2_base.py
│   │   │   └── sam2_utils.py
│   │   ├── sam2_hiera_b+.yaml
│   │   ├── sam2_hiera_l.yaml
│   │   ├── sam2_hiera_s.yaml
│   │   ├── sam2_hiera_t.yaml
│   │   ├── sam2_image_predictor.py
│   │   ├── sam2_video_predictor.py
│   │   └── utils
│   │       ├── __init__.py
│   │       ├── amg.py
│   │       ├── kalman_filter.py
│   │       ├── misc.py
│   │       └── transforms.py
│   ├── sav_dataset
│   │   ├── LICENSE
│   │   ├── LICENSE_DAVIS
│   │   ├── LICENSE_VOS_BENCHMARK
│   │   ├── README.md
│   │   ├── example
│   │   │   ├── sav_000001.mp4
│   │   │   ├── sav_000001_auto.json
│   │   │   └── sav_000001_manual.json
│   │   ├── requirements.txt
│   │   ├── sav_evaluator.py
│   │   ├── sav_visualization_example.ipynb
│   │   └── utils
│   │       ├── sav_benchmark.py
│   │       └── sav_utils.py
│   ├── setup.py
│   ├── tools
│   │   ├── README.md
│   │   └── vos_inference.py
│   └── training
│       ├── README.md
│       ├── __init__.py
│       ├── assets
│       │   ├── MOSE_sample_train_list.txt
│       │   └── MOSE_sample_val_list.txt
│       ├── dataset
│       │   ├── __init__.py
│       │   ├── sam2_datasets.py
│       │   ├── transforms.py
│       │   ├── utils.py
│       │   ├── vos_dataset.py
│       │   ├── vos_raw_dataset.py
│       │   ├── vos_sampler.py
│       │   └── vos_segment_loader.py
│       ├── loss_fns.py
│       ├── model
│       │   ├── __init__.py
│       │   └── sam2.py
│       ├── optimizer.py
│       ├── scripts
│       │   └── sav_frame_extraction_submitit.py
│       ├── train.py
│       ├── trainer.py
│       └── utils
│           ├── __init__.py
│           ├── checkpoint_utils.py
│           ├── data_utils.py
│           ├── distributed.py
│           ├── logger.py
│           └── train_utils.py
└── scripts
    ├── demo.py
    └── main_inference.py
```
<!-- END_STRUCTURE -->
</details>
