import os
import sys
from setuptools import setup, find_packages


readme = open('README.md').read()
setup(
    name='snippets',
    version='1.0.0',
    packages=find_packages(),
    description='Get access to all functions built by susmit',
    long_description=readme,
    author='Susmit Saha',
    author_email='susmit.saha@dream11.com',
    install_requires = ['pymysql==1.0.2', 
                        'boto3==1.14.19', 
                        'pyathena==2.3.0', 
                        'psycopg2==2.8.5', 
                        'requests', 
                        's3fs', 
                        'kafka-python', 
                        'tabulate==0.8.7', 
                        'gspread==3.6.0', 
                        'texttable==1.6.2', 
                        'slack-webhook==1.0.3', 
                        'mlflow', 
                        'django-environ', 
                        'numpy==1.23.1', 
                        'pandas', 
                        'scikit-learn', 
                        'pyspark', 
                        'scipy', 
                        'matplotlib', 
                        'joblib',
                        'delta-spark',
                        ],
    include_package_data=True,
)