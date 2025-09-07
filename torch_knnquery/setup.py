from setuptools import setup, find_packages
import torch
from torch.utils.cpp_extension import CUDAExtension, BuildExtension, CUDA_HOME

print('CUDA_HOME', CUDA_HOME)
if not torch.cuda.is_available() or CUDA_HOME is None:
    raise NotImplementedError('The torch_knnquery module only works with CUDA')

ext_modules = [
    CUDAExtension('knnquery_cuda',
                    ['src/knnquery.cu'])
]

__version__ = '1.0.0'

install_requires = ['torchvision']
tests_require = ['numpy']

setup(
    name='torch_knnquery',
    version=__version__,
    description='Implementation of voxel grid data structure for KNN point queries on the GPU in PyTorch',
    author='Jan Eric Lenssen',
    author_email='janeric.lenssen@tu-dortmund.de',
    keywords=[
        'pytorch', 'voxel grid', 'knn', 'neural points'
    ],
    install_requires=install_requires,
    tests_require=tests_require,
    ext_modules=ext_modules,
    cmdclass={
        'build_ext':
        BuildExtension
    },
    packages=find_packages(),
)