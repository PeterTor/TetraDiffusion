#!/usr/bin/env bash
#SBATCH -n 12
#SBATCH --time=120:00:00
#SBATCH --mem-per-cpu=7000

##SBATCH --gpus=4
##SBATCH --gres=gpumem:18288m

#SBATCH --gpus=rtx_4090:4
#SBATCH --job-name=car
#SBATCH -o ./car/slurm.out # STDOUT

name=car
id=02958343 #03790512 #02691156 #03790512 #02958343
thres=0
res=128
bs=4
ga=32
echo "Starting job $1"
echo "Starting to copy..."
mkdir $TMPDIR/shapenet_diffusion
mkdir $TMPDIR/shapenet_diffusion/preprocessed_data
rsync -a --exclude='*.obj' --exclude='*.txt' /cluster/work/igp_psr/nkalischek/meshprocessing_cluster/out/02958343.zip $TMPDIR/shapenet_diffusion/
echo "unzipping"
unzip -q -o $TMPDIR/shapenet_diffusion/\*.zip -d $TMPDIR/shapenet_diffusion/
echo "Done."

##accelerate launch --multi_gpu --num_processes 2 --gpu_ids "0,1" main_cluster.py --data_path $TMPDIR/shapenet_diffusion --grid_res $res --batch_size $bs
accelerate launch --multi_gpu --main_process_port 29519 --num_processes $bs --gpu_ids all main.py --data_path $TMPDIR/shapenet_diffusion --grid_res $res --batch_size $bs --name $name --shapenet_id $id --threshold $thres --load_weights 1 --ga $ga
