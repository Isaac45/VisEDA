"""
VisEDA — Visual Exploratory Data Analysis
==========================================

A Python library for performing rich EDA on image datasets,
hyperspectral data, point clouds, and videos.
"""

__version__ = "0.1.0"
__author__ = "Isaac Osei Agyemang", "Linda", "Stacy", "Martin", "Ijeoman"
__email__ = "ioa2@stir.ac.uk"
__license__ = "MIT"

from viseda.image import ImageEDA
from viseda.hyperspectral import HyperspectralEDA
from viseda.pointcloud import PointCloudEDA
from viseda.video import VideoEDA

__all__ = ["ImageEDA", "HyperspectralEDA", "PointCloudEDA", "VideoEDA"]
