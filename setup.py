import setuptools
import sys

with open("Readme.rst", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as f:
    install_requires = f.read().strip().split('\n')

# We raise an error if trying to install with python2
if sys.version[0] == '2':
    print("Error: This package must be installed with python3")
    sys.exit(1)

setuptools.setup(
    name="MManagement",
    version="2",
    author="Mateo Espitia-Ibarra",
    author_email="mespitiaibarra@gmail.com",
    description="Optimization tool for isolated microgrid management",
    long_description=long_description,
    long_description_content_type="text/x-rst",
    url="https://github.com/SENECA-UDEA/microgrids_management",
    packages=setuptools.find_packages(install_requires),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Operating System :: OS Independent",
    ],
    # install_requires=install_requires,
    include_package_data=True,
)
