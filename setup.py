"""
Installs the qdevplot package
"""

import pandas
from setuptools import setup, find_packages
from pathlib import Path

import versioneer

readme_file_path = Path(__file__).absolute().parent / "README.md"

required_packages = ['matplotlib>=3.0.0',
                     'numpy>=1.12.0',
                     'pandas'
                    ]
package_data = {"qdevplot": ["conf/telemetry.ini"] }


setup(
    name="qdevplot",
    version=versioneer.get_version(),
    cmdclass=versioneer.get_cmdclass(),
    python_requires=">=3.7",
    install_requires=required_packages,
    author= "Rasmus Bjerregaard Christensen",
    author_email="rbcmail@gmail.com",
    description="Plot functions for QDev",
    long_description=readme_file_path.open().read(),
    long_description_content_type="text/markdown",
    license="",
    package_data=package_data,
    packages=find_packages(exclude=["*.tests", "*.tests.*"]),
    classifiers=[
        "Development Status :: 2 - Pre-Alpha",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.7",
    ],
)
