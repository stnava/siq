
from setuptools import setup

long_description = open("README.md").read()

setup(name='antspydpr',
      version='0.0.0',
      description='deep perceptual resampling and super resolution with antspyx',
      long_description=long_description,
      long_description_content_type="text/markdown; charset=UTF-8; variant=GFM",
      url='https://github.com/stnava/ANTsPyDeepPerceptualResampling',
      author='Avants, Gosselin, Tustison, Reardon',
      author_email='stnava@gmail.com',
      license='Apache 2.0',
      packages=['antspydpr'],
      zip_safe=False)
