import argparse
import os
import os.path as osp
import numpy as np
import cv2
import torch
import gc
import sys
import logging
sys.path.append("./sam2")
from sam2.build_sam import build_sam2_video_predictor

logging.basicConfig(level=logging.INFO)

color = [(255, 0, 0)]

def load_txt(gt_path):
    with open(gt_path, 'r') as f:
        gt = f.readlines()
    prompts = {}
    for fid, line in enumerate(gt):
        try:
            x, y, w, h = map(float, line.split(','))
            x, y, w, h = int(x), int(y), int(w), int(h)
            prompts[fid] = ((x, y, x + w, y + h), 0)
        except ValueError:
            logging.warning(f"Skipping malformed line {fid} in {gt_path}")
    return prompts

def determine_model_cfg(model_path):
    if "large" in model_path:
        return "configs/samurai/sam2.1_hiera_l.yaml"
    elif "base_plus" in model_path:
        return "configs/samurai/sam2.1_hiera_b+.yaml"
    elif "small" in model_path:
        return "configs/samurai/sam2.1_hiera_s.yaml"
    elif "tiny" in model_path:
        return "configs/samurai/sam2.1_hiera_t.yaml"
    else:
        logging.warning("Unknown model size in path! Using default configuration.")
        return "configs/samurai/sam2.1_hiera_b+.yaml"

def prepare_frames_or_path(video_path):
    if video_path.endswith(('.mp4', '.avi', '.mov')) or osp.isdir(video_path):
        return video_path
    else:
        raise ValueError("Invalid video_path format. Should be .mp4, .avi, .mov or a directory of jpg frames.")

def main(args):
    model_cfg = determine_model_cfg(args.model_path)
    predictor = build_sam2_video_predictor(model_cfg, args.model_path, device="cuda:0")
    frames_or_path = prepare_frames_or_path(args.video_path)
    prompts = load_txt(args.txt_path)
    
    frame_rate = 30
    loaded_frames = []
    if args.save_to_video:
        if osp.isdir(args.video_path):
            frames = sorted([osp.join(args.video_path, f) for f in os.listdir(args.video_path) if f.lower().endswith((".jpg", ".jpeg"))])
            loaded_frames = [cv2.imread(frame_path) for frame_path in frames if cv2.imread(frame_path) is not None]
        else:
            cap = cv2.VideoCapture(args.video_path)
            frame_rate = cap.get(cv2.CAP_PROP_FPS)
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break
                loaded_frames.append(frame)
            cap.release()

    if not loaded_frames:
        raise ValueError("No frames were loaded. Please check the input video or image directory.")

    height, width = loaded_frames[0].shape[:2]
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(args.video_output_path, fourcc, frame_rate, (width, height))

    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.float16):
        state = predictor.init_state(frames_or_path, offload_video_to_cpu=True)
        bbox, track_label = prompts.get(0, ((0, 0, 0, 0), 0))
        _, _, masks = predictor.add_new_points_or_box(state, box=bbox, frame_idx=0, obj_id=0)

        for frame_idx, object_ids, masks in predictor.propagate_in_video(state):
            mask_to_vis = {}
            bbox_to_vis = {}

            for obj_id, mask in zip(object_ids, masks):
                mask = mask[0].cpu().numpy()
                mask = mask > 0.0
                non_zero_indices = np.argwhere(mask)
                bbox = [0, 0, 0, 0] if len(non_zero_indices) == 0 else [*non_zero_indices.min(axis=0), *(non_zero_indices.max(axis=0) - non_zero_indices.min(axis=0))]
                bbox_to_vis[obj_id] = bbox
                mask_to_vis[obj_id] = mask

            if args.save_to_video:
                img = loaded_frames[frame_idx]
                for obj_id, mask in mask_to_vis.items():
                    mask_img = np.zeros((height, width, 3), np.uint8)
                    mask_img[mask] = color[(obj_id + 1) % len(color)]
                    img = cv2.addWeighted(img, 1, mask_img, 0.2, 0)
                for obj_id, bbox in bbox_to_vis.items():
                    cv2.rectangle(img, tuple(bbox[:2]), tuple(bbox[2:]), color[obj_id % len(color)], 2)
                out.write(img)

        out.release()
    
    del predictor, state
    gc.collect()
    torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--video_path", required=True, help="Input video path or directory of frames.")
    parser.add_argument("--txt_path", required=True, help="Path to ground truth text file.")
    parser.add_argument("--model_path", default="sam2/checkpoints/sam2.1_hiera_base_plus.pt", help="Path to the model checkpoint.")
    parser.add_argument("--video_output_path", default="demo.mp4", help="Path to save the output video.")
    parser.add_argument("--save_to_video", type=bool, default=True, help="Save results to a video.")
    args = parser.parse_args()
    main(args)
