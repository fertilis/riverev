from setuptools import setup, find_packages

setup(
    name="riverev",
    version="1.0.0",
    packges=find_packages(),
    package_data={
        "riverev": ["*.so"],
    },
    install_requires=[
        "evio>=1.0.0",
    ],
)

