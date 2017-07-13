from setuptools import setup

setup(
    name='acp_instrument_sensitivity_function',
    version='0.1.1',
    description='Estimating the Integral Spectral Exclusion Zone of the ACP to calculate time-to-detections in the gamma-ray sky.',
    url='https://github.com/TheBigLebowSky/acp_instrument_sensitivity_function',
    author='Max Ludwig Ahnen, Sebastian Achim Mueller',
    author_email='m.knoetig@gmail.com',
    licence='GPL v3',
    packages=[
        'acp_instrument_sensitivity_function'
    ],
    package_data={'acp_instrument_sensitivity_function': [
        'resources/*',
        'resources/test_infolder/*'
        ]},
    entry_points={
        'console_scripts': [
            'acp_isez = acp_instrument_sensitivity_function.__main__:main'
        ]
    },
    install_requires=[
        'numpy',
        'scipy',
        'astropy',
        'matplotlib',
        'docopt',
        'tqdm',
        'gamma_limits_sensitivity'
    ],
    tests_require=['pytest']
)
