
# Just a little script to set up the python environment so it lives up to the PointPillar github repo requirements

#!/bin/bash
set -e  # stop on first error

# Make enviroment in relation to singularity image.
#srun singularity exec /ceph/project/P7_Gravel_Gun/pytorch_1.8.1-cuda11.1-cudnn8-devel.sif python -m venv --system-site-packages PointPillar_Env

# activate venv
source /ceph/project/P7_Gravel_Gun/PointPillar_Env/bin/activate

# upgrade pip + setuptools  # pip==21.3.1 is the latest version still fully compatible with Python 3.8 and older packaging tools from the PyTorch 1.8 / CUDA 11.1 era.
pip install --no-cache-dir --upgrade pip==21.3.1 setuptools==58.0.4 wheel

# install project dependencies
pip install --no-cache-dir \
  numba==0.48.0 \
  numpy==1.19.5 \
  open3d==0.14.1 \
  opencv-python==4.5.5.62 \
  PyYAML==6.0 \
  tensorboard \
  tqdm==4.62.3
