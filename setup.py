from setuptools import setup, find_packages

setup(
    name='neuralblock',
    version='0.1.0',
    package_dir={'': '.'}, # Look for packages in the current directory
    packages=find_packages(where='.'), # Find packages in the current directory
    include_package_data=True, # Include non-code files specified in MANIFEST.in
    install_requires=[
        "tflite-runtime",
        "numpy"
    ],
    # Metadata
    author='Jack Ellenberger',
    author_email='jellenberger@uchicago.edu',
    description='A package for identifying sponsored segments in transcripts.',
    url='https://github.com/jackellenberger/neuralblock-pkg',
)
