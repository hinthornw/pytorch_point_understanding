#!/bin/bash
today=`date '+%Y_%m_%d__%H_%M_%S'`;
filename="logs/${today}_training_log.txt"
cmd="python resnet_34_8s_train.py 2>&1 | tee ${filename}" 
echo ${cmd}
${cmd}
