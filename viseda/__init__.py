"""
VisEDA — Visual Exploratory Data Analysis
==========================================

A Python library for performing rich EDA on image datasets,
hyperspectral data, and point clouds.

Modules
-------
- viseda.image         : EDA for standard image datasets (RGB, grayscale)
- viseda.hyperspectral : EDA for hyperspectral / multispectral image cubes
- viseda.pointcloud    : EDA for 3D point cloud datasets
- viseda.report        : HTML report generation
"""

__version__ = "0.1.0"
__author__ = "Isaac Osei Agyemang", "Linda", "Stacy", "Martin", "Ijeoman"
__email__ = "ioa2@stir.ac.uk"
__license__ = "MIT"

from viseda.image import ImageEDA
from viseda.hyperspectral import HyperspectralEDA
from viseda.pointcloud import PointCloudEDA

__all__ = [
    "ImageEDA",
    "HyperspectralEDA",
    "PointCloudEDA",
]
