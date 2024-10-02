import os
import setuptools


setuptools.setup(
    name='twbench',
    version=open(os.path.join("twbench", "version.py")).readlines()[0].split("=")[-1].strip("' \n"),
    description='A benchmarking tool for language agents on interactive text environments.',
    url='https://github.com/microsoft/text-games-benchmark',
    long_description=open("README.md").read(),
    long_description_content_type='text/markdown',
    packages=setuptools.find_packages(),
    include_package_data=True,
    install_requires=open('requirements.txt').readlines(),
    classifiers=[
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],
)
