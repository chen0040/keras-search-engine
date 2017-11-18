from setuptools import setup

setup(
    name='keras_search_engine_web',
    packages=['keras_search_engine_web'],
    include_package_data=True,
    install_requires=[
        'flask',
        'keras',
        'sklearn',
        'nltk',
        'h5py'
    ],
    setup_requires=[
        'pytest-runner',
    ],
    tests_require=[
        'pytest',
    ],
)
