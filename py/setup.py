try:
    from setuptools import setup
except ImportError:
    from distutils.core import setup

config = {
    'description': 'Gao Wang\'s utility library',
    'author': 'Gao Wang',
    'version': '0.1',
    'packages': ['libgaow'],
    'package_dir': {'libgaow':'src'},
    'name': 'libgaow',
    'install_requires': ['scipy', 'scikit-learn', 'pandas']
}

setup(**config)
