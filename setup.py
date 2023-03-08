from setuptools import setup, find_packages

HTTPS_GITHUB_URL = "https://github.com/VLL-HD/FrEIA"

with open("README.rst", "r") as fh:
    long_description = fh.read()

setup(
    name="FrEIA",
    version="0.2",
    description="Framework for Easily Invertible Architectures",

    ## meta data
    author="VLL-HD team and collaborators",
    author_email="",
    long_description=long_description,
    long_description_content_type="text/rst",
    url=HTTPS_GITHUB_URL,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires='>=3.6',
    project_urls={  # Optional
        'Bug Reports': "https://github.com/VLL-HD/FrEIA"+"/issues",
        # 'Funding': 'https://donate.pypi.org',
        # 'Say Thanks!': 'http://saythanks.io/to/example',
        'Source': HTTPS_GITHUB_URL,
    },

    ## dependencies
    packages=find_packages(),
    install_requires=['numpy>=1.15.0','scipy>=1.5', 'torch>=1.0.0', 'graphviz>=0.20.1'],
    # extras_require={
    #     'testruns': ['pytest', 'pytest-benchmark'],
    # },
)
