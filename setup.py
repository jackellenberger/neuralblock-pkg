from setuptools import setup, find_packages

setup(
    name='neuralblock',
    version='0.1.0',
    packages=find_packages(),
    install_requires=[
        "tensorflow-cpu",
        "numpy"
    ],
    # Metadata
    author='Jack Ellenberger',
    author_email='jellenberger@uchicago.edu',
    description='A package for identifying sponsored segments in transcripts.',
    url='https://github.com/jackellenberger/neuralblock-pkg',
)
