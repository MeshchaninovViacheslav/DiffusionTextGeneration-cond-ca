#!/bin/bash

offline_runs="$1./offline-run-20230520_185426-4vxud*"
while :
do
    for ofrun in $offline_runs
    do
        wandb sync $ofrun;
    done
    sleep 1m
done