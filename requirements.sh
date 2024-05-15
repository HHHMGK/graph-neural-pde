TORCH=2.2.2
CUDA=cu121
pip install torch==2.2.2  --index-url https://download.pytorch.org/whl/cu121
pip install ogb pykeops
pip install torchdiffeq
pip install pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv -f https://data.pyg.org/whl/torch-${TORCH}+${CUDA}.html
pip install torch-geometric
pip install -r requirements2.txt