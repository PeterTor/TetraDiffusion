# ShapeNet preprocessing.
This repository provides code to fit an (ShapeNet) .obj file into the tetrahedral grid representation.

## 🚀 Run the script

    ```python
    python fit_single.py --shapenet /path/to/ShapeNetv2 --clid class_id --shid obj_id --out_dir /path/to/out

The script will run two rounds of optimization. In the second one, SDFs are fixed to -1, 1 and only the displacement vectors are optimized.
All parameters such as the grid size are loaded from `configs/shapenet.json`. 

Internally, `geometry/dmtet.py` does the heavy lifting. We use several loss functions. It might be necessary to change the weighting or remove some of the loss functions completely depending on the class.
