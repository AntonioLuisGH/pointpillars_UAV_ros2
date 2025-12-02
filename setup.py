from setuptools import setup, find_packages
import os
from glob import glob
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

package_name = 'pointpillars_ros'

setup(
    name=package_name,
    version='0.0.0',
    # We use find_packages to automatically find all sub-packages
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # Include the checkpoint file
        (os.path.join('share', package_name, 'models'), glob('models/*.pth')),
    ],
    install_requires=['setuptools'],
    zip_safe=False, # Must be False for C++ extensions
    maintainer='antonio',
    maintainer_email='antonio@todo.todo',
    description='PointPillars ROS2 Node',
    license='TODO',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'detect = pointpillars_ros.pointpillars_node:main',
        ],
    },
    # --- CUDA Extension Compilation ---
    ext_modules=[
        CUDAExtension(
            name='pointpillars_ros.drone_pointpillars.ops.voxel_op',
            sources=[
                'pointpillars_ros/drone_pointpillars/ops/voxelization/voxelization.cpp',
                'pointpillars_ros/drone_pointpillars/ops/voxelization/voxelization_cpu.cpp',
                'pointpillars_ros/drone_pointpillars/ops/voxelization/voxelization_cuda.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        ),
        CUDAExtension(
            name='pointpillars_ros.drone_pointpillars.ops.iou3d_op',
            sources=[
                'pointpillars_ros/drone_pointpillars/ops/iou3d/iou3d.cpp',
                'pointpillars_ros/drone_pointpillars/ops/iou3d/iou3d_kernel.cu',
            ],
            define_macros=[('WITH_CUDA', None)]
        )
    ],
    cmdclass={'build_ext': BuildExtension},
)