"""
Human-Only Enhanced ByteTrack Demo
====================================
Complete human-tracking pipeline with:

1. **Human-only detection** – filters YOLOX output to COCO class 0 (person)
   before anything reaches the tracker.  Configurable via ``--classes``.
2. **Enhanced tracker** – DIoU, confidence-aware matching, adaptive buffer.
3. **Tracking overlay** – tracked IDs rendered on output video.

Usage
-----
::

    # Full pipeline – human only:
    python tools/demo_track_human.py video ^
        -f exps/default/yolox_x.py ^
        -c pretrained/yolox_x.pth ^
        --path videos/palace.mp4 ^
        --fp16 --fuse --save_result

    # Track only class 0 and 1 (person + bicycle):
    python tools/demo_track_human.py video ^
        -f exps/default/yolox_x.py ^
        -c pretrained/yolox_x.pth ^
        --path videos/palace.mp4 ^
        --classes 0 1 --save_result
"""

import argparse
import os
import os.path as osp
import time
import cv2
import torch
import numpy as np
from loguru import logger

from yolox.data.data_augment import preproc
from yolox.exp import get_exp
from yolox.utils import fuse_model, get_model_info, postprocess
from yolox.tracker.enhanced_byte_tracker import EnhancedBYTETracker
from yolox.tracker.enhanced_visualize import plot_tracking_enhanced
from yolox.tracker.metrics_collector import MetricsCollector
from yolox.tracking_utils.timer import Timer


IMAGE_EXT = [".jpg", ".jpeg", ".webp", ".bmp", ".png"]

# COCO class names (80 classes) – index 0 = person
COCO_CLASSES = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep",
    "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard",
    "sports ball", "kite", "baseball bat", "baseball glove", "skateboard",
    "surfboard", "tennis racket", "bottle", "wine glass", "cup", "fork",
    "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair",
    "couch", "potted plant", "bed", "dining table", "toilet", "tv",
    "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave",
    "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
    "scissors", "teddy bear", "hair drier", "toothbrush",
]


# =====================================================================
# CLI argument parser
# =====================================================================
def make_parser():
    p = argparse.ArgumentParser("Human-Only Enhanced ByteTrack Demo")
    p.add_argument("demo", default="image",
                   help="demo type: image, video, webcam")
    p.add_argument("-expn", "--experiment-name", type=str, default=None)
    p.add_argument("-n", "--name", type=str, default=None, help="model name")
    p.add_argument("--path", default="./videos/palace.mp4",
                   help="path to images or video")
    p.add_argument("--camid", type=int, default=0)
    p.add_argument("--save_result", action="store_true",
                   help="save output video + MOT text file")

    # Model / experiment
    p.add_argument("-f", "--exp_file", default=None, type=str)
    p.add_argument("-c", "--ckpt", default=None, type=str)
    p.add_argument("--device", default="gpu", type=str)
    p.add_argument("--conf", default=None, type=float,
                   help="detection confidence threshold (e.g. 0.4)")
    p.add_argument("--nms", default=0.7, type=float,
                   help="NMS IoU threshold")
    p.add_argument("--tsize", default=800, type=int, help="test image size")
    p.add_argument("--fps", default=30, type=int, help="frame rate (fps)")
    p.add_argument("--fp16", default=False, action="store_true")
    p.add_argument("--fuse", default=False, action="store_true")
    p.add_argument("--trt", default=False, action="store_true")

    # ---- Human-only / class filtering -----------------------------------
    p.add_argument("--person_only", action="store_true", default=True,
                   help="Track only persons (COCO class 0). On by default.")
    p.add_argument("--classes", type=int, nargs="+", default=None,
                   help="COCO class IDs to track (e.g. --classes 0). "
                        "Overrides --person_only if given.")

    # ---- Tracking args ---------------------------------------------------
    p.add_argument("--track_thresh", type=float, default=0.3)
    p.add_argument("--track_buffer", type=int, default=60)
    p.add_argument("--match_thresh", type=float, default=0.9)
    p.add_argument("--aspect_ratio_thresh", type=float, default=2.5)
    p.add_argument("--min_box_area", type=float, default=5)
    p.add_argument("--mot20", default=False, action="store_true")

    # ---- Enhanced tracker flags ------------------------------------------
    p.add_argument("--no_cmc", action="store_true")
    p.add_argument("--no_reid", action="store_true")
    p.add_argument("--no_diou", action="store_true")
    p.add_argument("--no_adaptive_buf", action="store_true")
    p.add_argument("--no_occlusion", action="store_true")
    p.add_argument("--reid_model", type=str, default=None)
    p.add_argument("--lambda_iou", type=float, default=0.5)

    # ---- Metrics & graphs ------------------------------------------------
    p.add_argument("--save_metrics", action="store_true",
                   help="Save per-frame CSV + aggregate JSON metrics")
    p.add_argument("--save_graphs", action="store_true",
                   help="Generate and save presentation PNGs (requires matplotlib)")

    return p


# =====================================================================
# Class filter – applied to raw YOLOX output BEFORE tracker.update()
# =====================================================================
def filter_detections_by_class(output, allowed_classes):
    """
    Filter YOLOX post-processed output tensor to keep only allowed classes.

    Parameters
    ----------
    output : torch.Tensor  (N, 7)
        Columns: [x1, y1, x2, y2, obj_conf, class_conf, class_id]
    allowed_classes : set[int]
        Set of COCO class indices to keep.

    Returns
    -------
    filtered : torch.Tensor  (M, 7)
    """
    if output is None:
        return None
    # class_id is in column 6
    if isinstance(output, np.ndarray):
        class_ids = output[:, 6].astype(int)
        if allowed_classes == {0}:
            mask = (class_ids == 0)
        else:
            mask = np.isin(class_ids, list(allowed_classes))
        return output[mask]
    else:
        class_ids = output[:, 6].int()
        if allowed_classes == {0}:
            mask = (class_ids == 0)
        else:
            mask = torch.zeros(len(class_ids), dtype=torch.bool, device=output.device)
            for cid in allowed_classes:
                mask |= (class_ids == cid)
        return output[mask]


# =====================================================================
# Predictor (same as demo_track_enhanced but self-contained)
# =====================================================================
class Predictor:
    def __init__(self, model, exp, trt_file=None, decoder=None,
                 device=torch.device("cpu"), fp16=False):
        self.model = model
        self.decoder = decoder
        self.num_classes = exp.num_classes
        self.confthre = exp.test_conf
        self.nmsthre = exp.nmsthre
        self.test_size = exp.test_size
        self.device = device
        self.fp16 = fp16
        if trt_file is not None:
            from torch2trt import TRTModule
            model_trt = TRTModule()
            model_trt.load_state_dict(torch.load(trt_file))
            x = torch.ones((1, 3, exp.test_size[0], exp.test_size[1]),
                           device=device)
            self.model(x)
            self.model = model_trt
        self.rgb_means = (0.485, 0.456, 0.406)
        self.std = (0.229, 0.224, 0.225)

    def inference(self, img, timer):
        img_info = {"id": 0}
        if isinstance(img, str):
            img_info["file_name"] = osp.basename(img)
            img = cv2.imread(img)
        else:
            img_info["file_name"] = None

        height, width = img.shape[:2]
        img_info["height"] = height
        img_info["width"] = width
        img_info["raw_img"] = img

        img_prep, ratio = preproc(img, self.test_size, self.rgb_means, self.std)
        img_info["ratio"] = ratio
        img_t = torch.from_numpy(img_prep).unsqueeze(0).float().to(self.device)
        if self.fp16:
            img_t = img_t.half()

        with torch.no_grad():
            timer.tic()
            outputs = self.model(img_t)
            if self.decoder is not None:
                outputs = self.decoder(outputs, dtype=outputs.type())
            outputs = postprocess(outputs, self.num_classes,
                                  self.confthre, self.nmsthre)
        return outputs, img_info


# =====================================================================
# Helpers
# =====================================================================
def get_image_list(path):
    image_names = []
    for maindir, subdir, file_name_list in os.walk(path):
        for filename in file_name_list:
            apath = osp.join(maindir, filename)
            ext = osp.splitext(apath)[1]
            if ext in IMAGE_EXT:
                image_names.append(apath)
    return image_names


# =====================================================================
# Video demo loop
# =====================================================================
def imageflow_demo(predictor, vis_folder, current_time, args,
                   tracker, allowed_classes):
    cap = cv2.VideoCapture(args.path if args.demo == "video" else args.camid)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    if video_fps <= 0:
        video_fps = args.fps
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    if args.demo == "video":
        video_name = osp.splitext(osp.basename(args.path))[0]
        save_path = osp.join(save_folder, f"{video_name}_tracked.mp4")
    else:
        save_path = osp.join(save_folder, "camera.mp4")

    logger.info(f"Video save path: {save_path}")
    logger.info(f"Video: {width}x{height} @ {video_fps:.1f} FPS, ~{total_frames} frames")
    logger.info(f"Tracking classes: {sorted(allowed_classes)} "
                f"({', '.join(COCO_CLASSES[c] for c in sorted(allowed_classes) if c < len(COCO_CLASSES))})")

    vid_writer = cv2.VideoWriter(
        save_path, cv2.VideoWriter_fourcc(*"mp4v"),
        video_fps, (width, height))

    timer = Timer()
    collector = MetricsCollector()
    frame_id = 0
    results = []

    while True:
        ret_val, frame = cap.read()
        if not ret_val:
            break

        outputs, img_info = predictor.inference(frame, timer)

        raw_img = img_info["raw_img"]

        if outputs[0] is not None:
            # ---- CLASS FILTER (before tracker) ----------------------------
            filtered = filter_detections_by_class(outputs[0], allowed_classes)

            if filtered is not None and len(filtered) > 0:
                online_targets = tracker.update(
                    filtered,
                    [img_info["height"], img_info["width"]],
                    predictor.test_size,
                    img=raw_img,
                )
            else:
                online_targets = tracker.update(
                    np.empty((0, 5), dtype=np.float32),
                    [img_info["height"], img_info["width"]],
                    predictor.test_size,
                    img=raw_img,
                )
        else:
            online_targets = tracker.update(
                np.empty((0, 5), dtype=np.float32),
                [img_info["height"], img_info["width"]],
                predictor.test_size,
                img=raw_img,
            )

        online_tlwhs = []
        online_ids = []
        online_scores = []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                    f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )

        timer.toc()
        cur_fps = 1.0 / max(1e-5, timer.average_time)

        lost_count = len(tracker.lost_stracks) if hasattr(tracker, "lost_stracks") else 0
        collector.update(
            frame_id=frame_id,
            tlwhs=online_tlwhs,
            track_ids=online_ids,
            scores=online_scores,
            fps=cur_fps,
            lost_count=lost_count,
        )

        online_im = plot_tracking_enhanced(
            raw_img, online_tlwhs, online_ids,
            scores=online_scores,
            frame_id=frame_id + 1,
            fps=cur_fps,
            num_active_tracks=len(online_targets),
        )

        if args.save_result:
            vid_writer.write(online_im)

        if frame_id % 20 == 0:
            logger.info(
                f"Frame {frame_id}/{total_frames}  |  "
                f"{cur_fps:.1f} FPS  |  "
                f"{len(online_tlwhs)} humans  |  "
                f"{len(online_targets)} tracks"
            )

        ch = cv2.waitKey(1)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break
        frame_id += 1

    cap.release()
    vid_writer.release()

    # ---- Save MOT results -----------------------------------------------
    if args.save_result:
        res_file = osp.join(save_folder, f"{timestamp}_mot.txt")
        with open(res_file, "w") as f:
            f.writelines(results)
        logger.info(f"MOT results → {res_file}")

    collector.finalize()
    collector.print_summary()

    if args.save_metrics or args.save_graphs:
        csv_path = osp.join(save_folder, "metrics_per_frame.csv")
        json_path = osp.join(save_folder, "metrics_summary.json")
        collector.save_csv(csv_path)
        collector.save_json(json_path)
        logger.info(f"Metrics CSV  â†’ {csv_path}")
        logger.info(f"Metrics JSON â†’ {json_path}")

    if args.save_graphs:
        json_path = osp.join(save_folder, "metrics_summary.json")
        graphs_dir = osp.join(save_folder, "graphs")
        try:
            from yolox.tracker.graph_generator import generate_graphs
            generate_graphs(json_path, graphs_dir)
            logger.info(f"Graphs â†’ {graphs_dir}/")
        except Exception as e:
            logger.warning(f"Graph generation failed: {e}")

    logger.info("Done.")


# =====================================================================
# Image demo loop
# =====================================================================
def image_demo(predictor, vis_folder, current_time, args,
               tracker, allowed_classes):
    if osp.isdir(args.path):
        files = get_image_list(args.path)
    else:
        files = [args.path]
    files.sort()

    timestamp = time.strftime("%Y_%m_%d_%H_%M_%S", current_time)
    save_folder = osp.join(vis_folder, timestamp)
    os.makedirs(save_folder, exist_ok=True)

    timer = Timer()
    collector = MetricsCollector()
    results = []

    for frame_id, img_path in enumerate(files, 1):
        outputs, img_info = predictor.inference(img_path, timer)
        raw_img = img_info["raw_img"]

        if outputs[0] is not None:
            filtered = filter_detections_by_class(outputs[0], allowed_classes)
            if filtered is not None and len(filtered) > 0:
                online_targets = tracker.update(
                    filtered,
                    [img_info["height"], img_info["width"]],
                    predictor.test_size,
                    img=raw_img,
                )
            else:
                online_targets = tracker.update(
                    np.empty((0, 5), dtype=np.float32),
                    [img_info["height"], img_info["width"]],
                    predictor.test_size,
                    img=raw_img,
                )
        else:
            online_targets = tracker.update(
                np.empty((0, 5), dtype=np.float32),
                [img_info["height"], img_info["width"]],
                predictor.test_size,
                img=raw_img,
            )

        online_tlwhs, online_ids, online_scores = [], [], []
        for t in online_targets:
            tlwh = t.tlwh
            tid = t.track_id
            vertical = tlwh[2] / tlwh[3] > args.aspect_ratio_thresh
            if tlwh[2] * tlwh[3] > args.min_box_area and not vertical:
                online_tlwhs.append(tlwh)
                online_ids.append(tid)
                online_scores.append(t.score)
                results.append(
                    f"{frame_id},{tid},{tlwh[0]:.2f},{tlwh[1]:.2f},"
                    f"{tlwh[2]:.2f},{tlwh[3]:.2f},{t.score:.2f},-1,-1,-1\n"
                )

        timer.toc()
        cur_fps = 1.0 / max(1e-5, timer.average_time)

        collector.update(frame_id, online_tlwhs, online_ids,
                         online_scores, cur_fps)

        online_im = plot_tracking_enhanced(
            raw_img, online_tlwhs, online_ids,
            scores=online_scores, frame_id=frame_id, fps=cur_fps)

        if args.save_result:
            cv2.imwrite(osp.join(save_folder, osp.basename(img_path)), online_im)

        if frame_id % 20 == 0:
            logger.info(f"Frame {frame_id}  |  {cur_fps:.1f} FPS  |  "
                        f"{len(online_tlwhs)} humans")

        ch = cv2.waitKey(0)
        if ch == 27 or ch == ord("q") or ch == ord("Q"):
            break

    # Save outputs
    if args.save_result:
        res_file = osp.join(save_folder, f"{timestamp}_mot.txt")
        with open(res_file, "w") as f:
            f.writelines(results)
        logger.info(f"MOT results → {res_file}")


    collector.finalize()
    collector.print_summary()

    if args.save_metrics or args.save_graphs:
        csv_path = osp.join(save_folder, "metrics_per_frame.csv")
        json_path = osp.join(save_folder, "metrics_summary.json")
        collector.save_csv(csv_path)
        collector.save_json(json_path)

    if args.save_graphs:
        json_path = osp.join(save_folder, "metrics_summary.json")
        graphs_dir = osp.join(save_folder, "graphs")
        try:
            from yolox.tracker.graph_generator import generate_graphs
            generate_graphs(json_path, graphs_dir)
        except Exception as e:
            logger.warning(f"Graph generation failed: {e}")

# =====================================================================
# Main
# =====================================================================
def main(exp, args):
    if not args.experiment_name:
        args.experiment_name = exp.exp_name

    output_dir = osp.join(exp.output_dir, args.experiment_name)
    os.makedirs(output_dir, exist_ok=True)

    vis_folder = osp.join(output_dir, "track_vis")
    os.makedirs(vis_folder, exist_ok=True)

    if args.trt:
        args.device = "gpu"
    args.device = torch.device("cuda" if args.device == "gpu" else "cpu")

    logger.info("Args: {}".format(args))

    # ---- Detection thresholds -------------------------------------------
    if args.conf is not None:
        exp.test_conf = args.conf
    if args.nms is not None:
        exp.nmsthre = args.nms
    if args.tsize is not None:
        exp.test_size = (args.tsize, args.tsize)

    logger.info(f"Detection conf threshold: {exp.test_conf}")
    logger.info(f"NMS threshold: {exp.nmsthre}")
    logger.info(f"Test size: {exp.test_size}")

    # ---- Model loading ---------------------------------------------------
    model = exp.get_model().to(args.device)
    logger.info("Model Summary: {}".format(get_model_info(model, exp.test_size)))
    model.eval()

    if not args.trt:
        ckpt_file = args.ckpt if args.ckpt else osp.join(output_dir, "best_ckpt.pth.tar")
        logger.info(f"Loading checkpoint: {ckpt_file}")
        ckpt = torch.load(ckpt_file, map_location="cpu")
        model.load_state_dict(ckpt["model"])
        logger.info("Checkpoint loaded.")

    if args.fuse:
        logger.info("Fusing model...")
        model = fuse_model(model)
    if args.fp16:
        model = model.half()

    if args.trt:
        assert not args.fuse
        trt_file = osp.join(output_dir, "model_trt.pth")
        assert osp.exists(trt_file)
        model.head.decode_in_inference = False
        decoder = model.head.decode_outputs
    else:
        trt_file = None
        decoder = None

    predictor = Predictor(model, exp, trt_file, decoder, args.device, args.fp16)

    # ---- Determine which classes to track --------------------------------
    if args.classes is not None:
        allowed_classes = set(args.classes)
    elif args.person_only:
        allowed_classes = {0}
    else:
        allowed_classes = set(range(80))  # all COCO classes

    class_names = [COCO_CLASSES[c] for c in sorted(allowed_classes)
                   if c < len(COCO_CLASSES)]
    logger.info(f"Tracking classes: {sorted(allowed_classes)} → {class_names}")

    # ---- Create enhanced tracker -----------------------------------------
    tracker = EnhancedBYTETracker(
        args,
        frame_rate=args.fps,
        enable_cmc=not args.no_cmc,
        enable_reid=not args.no_reid,
        enable_diou=not args.no_diou,
        enable_adaptive_buf=not args.no_adaptive_buf,
        enable_occlusion=not args.no_occlusion,
        reid_model_path=args.reid_model,
        lambda_iou=args.lambda_iou,
    )

    # ---- Run demo --------------------------------------------------------
    current_time = time.localtime()
    if args.demo == "image":
        image_demo(predictor, vis_folder, current_time, args,
                   tracker, allowed_classes)
    elif args.demo in ("video", "webcam"):
        imageflow_demo(predictor, vis_folder, current_time, args,
                       tracker, allowed_classes)


if __name__ == "__main__":
    args = make_parser().parse_args()
    exp = get_exp(args.exp_file, args.name)
    main(exp, args)
