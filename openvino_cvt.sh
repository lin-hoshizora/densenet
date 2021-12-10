#!/bin/bash
if [ $# -ne 5 ]
then
  echo "Usage: ./openvino_cvt.sh <TF_SAVED_MODEL_DIR> <PRECISION> <OUTPUT_DIR> <OUTPUT_MODEL_NAME> <DEV>"
  exit 1
fi

mo.py --batch 1 --saved_model_dir $1 --output_dir $3 --model_name $4 --data_type $2 && \
python3 openvino_verify.py $1 $3/$4 $5
