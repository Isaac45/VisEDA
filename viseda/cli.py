"""
viseda.cli
----------
Command-line interface for VisEDA.

Usage
-----
  viseda image  <path> [--max N] [--report report.html] [--plot]
  viseda hyper  <path> [--wavelengths wl.npy] [--plot]
  viseda cloud  <path> [--max N] [--plot]
"""

from __future__ import annotations

import argparse
import sys


def _cmd_image(args):
    from viseda import ImageEDA

    eda = ImageEDA(verbose=True, max_images=args.max)
    eda.load(args.path, label_from_parent=args.label_from_parent)
    s = eda.summary()
    _print_summary(s)
    if args.report:
        eda.report(args.report)
    if args.plot:
        eda.plot(save_path=args.save_plot)


def _cmd_hyper(args):
    import numpy as np
    from viseda import HyperspectralEDA

    wl = np.load(args.wavelengths) if args.wavelengths else None
    eda = HyperspectralEDA(verbose=True, wavelengths=wl)
    eda.load(args.path)
    s = eda.summary()
    _print_summary(s)
    if args.plot:
        eda.plot(save_path=args.save_plot)


def _cmd_cloud(args):
    from viseda import PointCloudEDA

    eda = PointCloudEDA(verbose=True, max_points=args.max)
    eda.load(args.path)
    s = eda.summary()
    _print_summary(s)
    if args.plot:
        eda.plot(save_path=args.save_plot)


def _print_summary(s):
    import json
    print(json.dumps(s, indent=2, default=str))


def main():
    parser = argparse.ArgumentParser(
        prog="viseda",
        description="VisEDA – Visual Exploratory Data Analysis",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # image sub-command
    p_img = sub.add_parser("image", help="EDA on an image dataset")
    p_img.add_argument("path", help="Directory or file path")
    p_img.add_argument("--max", type=int, default=None,
                       help="Max images to analyse")
    p_img.add_argument("--label-from-parent", action="store_true",
                       help="Use parent folder name as label")
    p_img.add_argument("--report", default=None,
                       help="Save HTML report to this path")
    p_img.add_argument("--plot", action="store_true",
                       help="Show / save dashboard")
    p_img.add_argument("--save-plot", default=None,
                       help="Save plot to this path instead of showing")

    # hyperspectral sub-command
    p_hs = sub.add_parser("hyper", help="EDA on a hyperspectral cube")
    p_hs.add_argument("path", help="Path to cube file (.hdr, .npy, .tif)")
    p_hs.add_argument("--wavelengths", default=None,
                      help="Path to .npy array of wavelengths (nm)")
    p_hs.add_argument("--plot", action="store_true")
    p_hs.add_argument("--save-plot", default=None)

    # pointcloud sub-command
    p_pc = sub.add_parser("cloud", help="EDA on a point cloud")
    p_pc.add_argument("path", help="Path to point cloud file")
    p_pc.add_argument("--max", type=int, default=1_000_000,
                      help="Max points to load")
    p_pc.add_argument("--plot", action="store_true")
    p_pc.add_argument("--save-plot", default=None)

    args = parser.parse_args()

    dispatch = {
        "image": _cmd_image,
        "hyper": _cmd_hyper,
        "cloud": _cmd_cloud,
    }
    dispatch[args.command](args)


if __name__ == "__main__":
    main()