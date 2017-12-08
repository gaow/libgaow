import sys
_py_ver = sys.version_info
if _py_ver.major == 2 or (_py_ver.major == 3 and (_py_ver.minor, _py_ver.micro) < (6, 0)):
    raise SystemError('Python 3.6 or higher is required')

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
    'install_requires': ['scipy', 'scikit-learn', 'pandas', 'pathos']
}

setup(**config)
