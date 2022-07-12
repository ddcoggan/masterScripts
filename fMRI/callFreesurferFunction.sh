#!/bin/bash

while getopts s: option
do
case "${option}"
in
s) STRING=${OPTARG};;
esac
done

FSLDIR=/usr/local/fsl
. ${FSLDIR}/etc/fslconf/fsl.sh
PATH=${FSLDIR}/bin:${PATH}
export FSLDIR PATH

export FREESURFER_HOME=/mnt/HDD12TB/freesurfer
export SUBJECTS_DIR=/mnt/HDD12TB/freesurfer/subjects
source $FREESURFER_HOME/SetUpFreeSurfer.sh

$STRING
