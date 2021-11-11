import os
import sys
from setuptools import setup, find_packages

# Check if Auto-sklearn could run on the given system
if os.name != 'posix':
    raise ValueError(
        'Detected unsupported operating system: %s. Please check '
        'the compability information of auto-sklearn: https://automl.github.io'
        '/auto-sklearn/master/installation.html#windows-osx-compatibility' %
        sys.platform
    )

if sys.version_info < (3, 6):
    raise ValueError(
        'Unsupported Python version %d.%d.%d found. Auto-sklearn requires Python '
        '3.6 or higher.' % (sys.version_info.major, sys.version_info.minor, sys.version_info.micro)
    )

HERE = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(HERE, 'requirements.txt')) as fp:
    install_reqs = [r.rstrip() for r in fp.readlines()
                    if not r.startswith('#') and not r.startswith('git+')]

with open('README.md') as fh:
    long_description = fh.read()

extras_reqs={
    "test": ["pytest", "pytest-cov"],
    "docs": [
        "sphinx",
        "sphinx-gallery",
        "sphinx_bootstrap_theme",
        "sphinx_rtd_theme",
        "numpydoc"
    ]
}

setup(
    name='SEAL',
    version='0.1.0',
    author='Taha Hessane, Pierre-Arthur Claudé',
    author_email='javslab@gmail.com',
    description='Design / automation of your machine learning experiment',
    long_description=long_description,
    long_description_content_type='text/markdown',
    packages=find_packages(),
    extras_require=extras_reqs,
    install_requires=install_reqs,
    platforms=['Linux'],
    python_requires='>=3.6',
    url='https://github.com/javslab/SEAL'
)
