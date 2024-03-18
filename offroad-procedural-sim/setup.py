import os
import pathlib

from setuptools import find_packages, setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / "README.md").read_text(encoding="utf-8")


def read_requirements_file(filename):
    req_file_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), filename)
    with open(req_file_path) as f:
        return [line.strip() for line in f]


description = "Procedural driving simulation in MuJoCo for reinforcement learning"

install_requires = read_requirements_file("requirements.txt")

setup(
    name="procedural_driving",
    version="0.0.1",
    description=description,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/kylestach/procedural_driving",
    author="Kyle Stachowicz",
    keywords="",
    packages=find_packages(),
    install_requires=install_requires,
    license="MIT",
)
