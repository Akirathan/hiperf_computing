#!/bin/bash

PARTITION="small-lp"

./move_files.sh
echo "==== Running remote compilation with $PARTITION partition... ===="

ssh parlab "cd levenstein_hiperf ; srun -p $PARTITION make"
