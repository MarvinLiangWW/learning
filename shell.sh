#!/bin/bash
# bash shell.sh --task 'gmv' --bs 20
# how to write argparse in bash
while [$# -gt 0]; do
  case "$1" in 
    --task) 
      shift
      task=$1
      shift
      ;;
    --bs)
      shift
      bs=$1
      echo bs=${bs}
      shift
      ;;
    *)
      other_arg=$1
      echo other_arg=${other_arg}
      shift
      ;;
  esac
done

      
