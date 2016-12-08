from setuptools import setup

setup(
    name='acp_paper_analysis',
    version='0.1',
    description='High level analysis & plotting methods for first ACP paper',
    url='https://github.com/TheBigLebowSky/acp_paper_analysis',
    author='Max Ahnen',
    author_email='m.knoetig@gmail.com',
    licence='MIT',
    packages=[
        'acp_paper_analysis'
    ],
    package_data={'acp_paper_analysis': ['resources/*']},
    entry_points={
        'console_scripts': [
            'acp_paper_analysis = acp_paper_analysis.__main__:main'
        ]
    },
    install_requires=[
        'numpy',
        'scipy',
        'matplotlib',
        'docopt',
        'pyfits'
        'gamma_limits_sensitivity'
    ],
    tests_require=['pytest']
)
