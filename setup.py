from setuptools import setup

with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()

__version__ = "0.0.0"

REPO_NAME = "kidney-disease-classification-dlops"
AUTHOR_USER_NAME = "Gargeesharmaa"
SRC_REPO = "Kidney Disease Classification"
AUTHOR_EMAIL = "gargeesharma52@gmail.com"



setup(
    name="kidney-disease-classification",
    version="0.0.0",
    author="Gargeesharmaa",
    description="A small kidney disease classification package",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=f"https://github.com/{AUTHOR_USER_NAME}/{REPO_NAME}",
    packages_dir={"":"src"},
    packages=setup.find_packages(),

)

