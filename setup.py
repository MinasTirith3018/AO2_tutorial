"""
This is a small package prepared by Yoonsoo P. Bach as a TA of
Astronomical Observation and Lab 2 course at Seoul National University.

version 0.0.1 2017-09-26

"""

from setuptools import setup

setup(
    name = "AO2tutorial",
    version = "0.0.1",
    author = "Yoonsoo Bach",
    author_email = "dbstn95@gmail.com",
    description = "",
    license = "",
    keywords = "",
    url = "",
    packages=['AO2tutorial'],
    requires=['numpy', 'scipy', 'astropy', 'ccdproc', 'matplotlib'],
)
