from tensortools import __version__
from setuptools import find_packages, setup

version = __version__

REQUIRED_PACKAGES = [
    'tensorflow >= 1.2.1',
    'scipy >= 0.19.1',
]

setup(
    name='tensortools',
    version=version,
    url='https://github.com/TheLampshady/tbd',
    author='Lampshady',
    author_email='lampshady24@gmail.com',
    description='A Python library for building simple Neural Networks '
                'and formatting input for networks',
    license='MIT',
    packages=find_packages(exclude=['tests']),
    install_requires=REQUIRED_PACKAGES,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Build Tools',
        'License :: OSI Approved :: MIT License',
        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
    ],
    keywords='tensorflow tensor machine learning',
)
