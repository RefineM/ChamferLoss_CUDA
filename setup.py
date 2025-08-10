from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='chamferloss_cuda',
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
