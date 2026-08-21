"""
VisEDA — Visual Exploratory Data Analysis
==========================================

A Python library for performing rich EDA on image datasets,
hyperspectral data, point clouds, videos, and text/NLP datasets.
"""

__version__ = "1.0.0"
__author__ = "Isaac Osei Agyemang"
__email__ = "agyemangisaac45@gmail.com"
__license__ = "MIT"

from viseda.image import ImageEDA
from viseda.hyperspectral import HyperspectralEDA
from viseda.pointcloud import PointCloudEDA
from viseda.video import VideoEDA
from viseda.text import TextEDA

__all__ = ["ImageEDA", "HyperspectralEDA", "PointCloudEDA", "VideoEDA", "TextEDA"]
