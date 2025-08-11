from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamferloss_cuda',
    author='Junfan Wang',
    author_email='1215718263@qq.com',
    version='1.0.0',
    ext_modules=[
        CUDAExtension(
            name='chamferloss_cuda',
            sources=['csrc/chamfer.cpp', 'csrc/kernel.cu'],
            extra_compile_args={'cxx': ['-O3'], 
                                'nvcc': ['-O3', '-lineinfo']}
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)
