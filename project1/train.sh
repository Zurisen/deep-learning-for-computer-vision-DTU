#!/bin/sh
#BSUB -q 02514
#BSUB -J model1
#BSUB -n 1
#BSUB -W 60
#BSUB -R "rusage[mem=16GB]"
#BSUB -o logs/log_%J.out
#BSUB -e logs/log_%J.err

mkdir -p logs
pip3 install pandas --user
pip3 install numpy --user
pip3 install torch --user
pip3 install matplotlib --user
pip3 install tdqm --user
pip3 install pillow  --user

python3 main_10.py

