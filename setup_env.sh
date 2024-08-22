## You need to first give a name of your environment

conda install pytorch==2.3.0 pytorch-cuda=12.1 -c pytorch -c nvidia
pip install torch_geometric
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-2.3.0+cu121.html
pip install rdkit==2024.03.1
pip install tensorboardX
pip install einops
pip install ipykernel

python -m pip install -e .[chem]

pip -y uninstall numpy
pip install numpy==1.26.4
pip install pandas


## Legacy Support of Development Environment
 
# As our project has been in development for over two years, the PyTorch ecosystem has undergone significant changes. To facilitate replication and extension of our work, we are providing the details of our original development environment as a point of reference.

# To restore the original development environment used in our project, please execute the commands listed below:
# conda install pytorch==1.12.1 torchvision==0.13.1 torchaudio==0.12.1 cudatoolkit=11.3 -c pytorch -y
# conda install pyg -c pyg -y
# conda install pytorch-cluster -c pyg -y
# pip install rdkit-pypi
# pip install tensorboardX
# pip install einops
# pip install ipykernel

# python -m pip install -e .[chem]
