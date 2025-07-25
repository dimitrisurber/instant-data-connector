"""Setup script for Instant Data Connector."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text() if (this_directory / "README.md").exists() else ""

# Read requirements
requirements = []
requirements_file = this_directory / "requirements.txt"
if requirements_file.exists():
    with open(requirements_file) as f:
        requirements = [line.strip() for line in f if line.strip() and not line.startswith("#")]

setup(
    name="instant-data-connector",
    version="0.1.0",
    author="Dimitri Surber",
    author_email="dimitri@ikigai2.com",
    description="ML-optimized data aggregation and serialization for instant algorithm development",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/dimitrisurber/instant-data-connector",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=requirements,
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "black>=22.0.0",
            "flake8>=5.0.0",
            "mypy>=0.990",
        ],
    },
    entry_points={
        "console_scripts": [
            "instant-aggregate=scripts.aggregate_data:main",
            "instant-load=scripts.load_data:main",
        ],
    },
    include_package_data=True,
    package_data={
        "instant_connector": ["py.typed"],
    },
)