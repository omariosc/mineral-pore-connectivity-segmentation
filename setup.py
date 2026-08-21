from setuptools import setup, find_packages

setup(
    name="mineral-pore-segmentation",
    version="0.1.0",
    description="Automated pipeline for segmenting and classifying pores in rock microscopy images",
    packages=find_packages(include=["src", "src.*", "config", "config.*"]),
    python_requires=">=3.11",
    install_requires=[
        "torch>=2.0.0",
        "torchvision",
        "numpy",
        "opencv-python",
        "scikit-image",
        "scikit-learn",
        "matplotlib",
        "pandas",
        "pycocotools",
        "pillow",
        "tqdm",
        "pyyaml",
        "click",
        "scipy",
        "albumentations==2.0.8",
    ],
    extras_require={
        "dev": [
            "pytest",
            "pytest-cov",
            "black",
            "flake8",
        ]
    },
)
