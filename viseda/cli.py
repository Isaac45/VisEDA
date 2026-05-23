"""viseda.cli - Command-line interface for VisEDA."""
from __future__ import annotations
import argparse


def _cmd_image(args):
    from viseda import ImageEDA
    eda = ImageEDA(verbose=True, max_images=args.max)
    eda.load(args.path, label_from_parent=args.label_from_parent)
    _print_summary(eda.summary())
    if args.report: eda.report(args.report)
    if args.plot: eda.plot(save_path=args.save_plot)


def _cmd_hyper(args):
    import numpy as np
    from viseda import HyperspectralEDA
    wl = np.load(args.wavelengths) if args.wavelengths else None
    eda = HyperspectralEDA(verbose=True, wavelengths=wl)
    eda.load(args.path, label_from_parent=args.label_from_parent)
    _print_summary(eda.summary())
    if args.report: eda.report(args.report)
    if args.plot:
        eda.plot_dataset(save_path=args.save_plot) if args.dataset_plot else eda.plot(save_path=args.save_plot)


def _cmd_cloud(args):
    from viseda import PointCloudEDA
    eda = PointCloudEDA(verbose=True, max_clouds=args.max_clouds, max_points_per_cloud=args.max_points)
    eda.load(args.path, label_from_parent=args.label_from_parent)
    _print_summary(eda.summary())
    if args.report: eda.report(args.report)
    if args.plot: eda.plot_dataset(save_path=args.save_plot)


def _cmd_video(args):
    from viseda import VideoEDA
    eda = VideoEDA(
        verbose=True,
        max_videos=args.max,
        frame_sample_rate=args.frame_sample_rate,
        max_frames_per_video=args.max_frames_per_video,
        blur_threshold=args.blur_threshold,
        motion_blur_threshold=args.motion_blur_threshold,
    )
    eda.load(args.path, label_from_parent=args.label_from_parent)
    _print_summary(eda.summary())
    if args.report: eda.report(args.report)
    if args.plot:
        eda.plot(video_index=0, save_path=args.save_plot) if args.single_plot else eda.plot_dataset(save_path=args.save_plot)


def _print_summary(s):
    import json
    print(json.dumps(s, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(prog="viseda", description="VisEDA – Visual Exploratory Data Analysis")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("image", help="EDA on an image dataset")
    p.add_argument("path"); p.add_argument("--max", type=int, default=None)
    p.add_argument("--label-from-parent", action="store_true"); p.add_argument("--report", default=None)
    p.add_argument("--plot", action="store_true"); p.add_argument("--save-plot", default=None)

    p = sub.add_parser("hyper", help="EDA on a hyperspectral cube or dataset")
    p.add_argument("path"); p.add_argument("--wavelengths", default=None)
    p.add_argument("--label-from-parent", action="store_true"); p.add_argument("--report", default=None)
    p.add_argument("--plot", action="store_true"); p.add_argument("--dataset-plot", action="store_true")
    p.add_argument("--save-plot", default=None)

    p = sub.add_parser("cloud", help="EDA on a point cloud dataset")
    p.add_argument("path"); p.add_argument("--max-clouds", type=int, default=None)
    p.add_argument("--max-points", type=int, default=1_000_000)
    p.add_argument("--label-from-parent", action="store_true"); p.add_argument("--report", default=None)
    p.add_argument("--plot", action="store_true"); p.add_argument("--save-plot", default=None)

    p = sub.add_parser("video", help="EDA on a single video or video dataset")
    p.add_argument("path"); p.add_argument("--max", type=int, default=None)
    p.add_argument("--frame-sample-rate", type=int, default=5)
    p.add_argument("--max-frames-per-video", type=int, default=300)
    p.add_argument("--blur-threshold", type=float, default=80.0)
    p.add_argument("--motion-blur-threshold", type=float, default=80.0)
    p.add_argument("--label-from-parent", action="store_true"); p.add_argument("--report", default=None)
    p.add_argument("--plot", action="store_true"); p.add_argument("--single-plot", action="store_true")
    p.add_argument("--save-plot", default=None)

    args = parser.parse_args()
    {"image": _cmd_image, "hyper": _cmd_hyper, "cloud": _cmd_cloud, "video": _cmd_video}[args.command](args)


if __name__ == "__main__":
    main()
