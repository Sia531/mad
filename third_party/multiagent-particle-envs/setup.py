from setuptools import find_packages, setup

setup(
    name="multiagent",
    version="0.0.1",
    description="Multi-Agent Goal-Driven Communication Environment",
    url="https://github.com/openai/multiagent-public",
    author="Igor Mordatch",
    author_email="mordatch@openai.com",
    package_data={"multiagent": ["secrcode.ttf"]},
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    install_requires=["gym", "numpy-stl"],
)
