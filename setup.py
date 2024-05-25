import setuptools

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setuptools.setup(
    name="aswtunner",
    version="0.0.2",
    author="loc nguyen tan",
    author_email="Loc.Nguyen.Tan@aswatson.com ",
    description="Wrapper for tunning machine learning model",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://dev.azure.com/ASWBigData/GSP-DataScience-Personalization/_git/asw-tunner",
    project_urls={
        "Bug Tracker": "https://dev.azure.com/ASWBigData/GSP-DataScience-Personalization/_git/asw-tunner/-/issues",
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    install_requires=required
)
