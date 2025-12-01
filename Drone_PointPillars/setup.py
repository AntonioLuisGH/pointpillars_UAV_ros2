from setuptools import setup, find_packages
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='drone_pointpillars',
    version='0.1',
    packages=find_packages(),
    ext_modules=[
        CUDAExtension(
            name='drone_pointpillars.ops.voxel_op',
            sources=[
                'drone_pointpillars/ops/voxelization/voxelization.cpp',
                'drone_pointpillars/ops/voxelization/voxelization_cpu.cpp',
                'drone_pointpillars/ops/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='drone_pointpillars.ops.iou3d_op',
            sources=[
                'drone_pointpillars/ops/iou3d/iou3d.cpp',
                'drone_pointpillars/ops/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
    zip_safe=False
)