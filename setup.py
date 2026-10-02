# Modified for ReSAIL. See NOTICE and LICENSE for attribution and terms.
from setuptools import find_packages, setup


def _fetch_requirements(path):
    with open(path) as fd:
        return [r.strip() for r in fd.readlines() if r.strip() and not r.startswith("#")]


# Setup configuration
setup(
    author="ReSAIL contributors and slime Team",
    name="resail",
    version="0.1.0",
    description="Retentive and Selective Augmentation for Iterative Self-Distillation",
    packages=find_packages(include=["slime*", "slime_plugins*"]),
    include_package_data=True,
    install_requires=_fetch_requirements("requirements.txt"),
    extras_require={},
    python_requires=">=3.10",
    classifiers=[
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Environment :: GPU :: NVIDIA CUDA",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Distributed Computing",
    ],
)
