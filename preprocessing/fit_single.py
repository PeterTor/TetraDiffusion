"""
Fit single ShapeNet obj into tetrahedral representation.
Config is taken from configs/shapenet.json. Change shapenet.json for different settings, such as resolution.
"""
import json
import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--shapenet", type=str, default="/scratch4/meshdiffusion/ShapeNetCore.v2")
parser.add_argument("--clid", type=str, default="03790512")
parser.add_argument("--shid", type=str, default="fb3e72a17d45156eef7c0039cbd1871c")
parser.add_argument("--out_dir", type=str, default="./meshprocessing_cluster/out")
args = parser.parse_args()

clid = args.clid
shid = args.shid
o_j = json.load(open("./configs/shapenet.json"))

shape = f"{args.shapenet}/{clid}/{shid}/models/model_normalized.obj"

o_j["ref_mesh"] = shape
o_j["mtl_override"] = shape.replace(".obj",".mtl")
o_j["out_dir"] = f"{clid}/{shid}"
path = f"./configs/{clid}/"
os.makedirs(path, exist_ok=True)

with open(f"{path}{shid}.json", "w") as outfile:
    json.dump(o_j, outfile)

if not os.path.exists(f"{args.out_dir}/{clid}/{shid}/mesh_data/mesh.obj"):
    os.system(f"python train.py --config {path}{shid}.json")

