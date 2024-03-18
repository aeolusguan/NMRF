from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(name='downsample_sp',
      ext_modules=[
          CUDAExtension('downsample_sp', [
              'downsample.cpp',
              'downsample_kernel.cu',
          ])
      ],
      cmdclass={'build_ext': BuildExtension})