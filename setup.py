from setuptools import setup
import os

try:
   import pypandoc
   long_description = pypandoc.convert('README.md', 'rst')
except (IOError, ImportError):
   long_description = open('README.md').read()

setup(name='demystipy',
      version='0.0.1',
      description='Identify interesting columns in a dataset',
      long_description=long_description,
      url='https://github.com/popily/demystipy',
      download_url ='https://github.com/popily/demystipy/tarball/0.0.1',
      author='Jonathon Morgan',
      author_email='jonathon@popily.com',
      license='MIT',
      packages=['demystipy'],
      test_suite='tests',
      install_requires=['sklearn','numpy','pandas'],
      zip_safe=False)