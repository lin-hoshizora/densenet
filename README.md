## TensorFlow
* TensorFlow inference & saved model: tf-demo.ipynb
* TF-Lite: tflite/tflite.ipynb

## OpenVINO
* TF saved modelからOpenVINOモデル(FP32)へ変換し、MYRIADで精度ロスを確認
````
./openvino_cvt.sh saved_models/crnn192 FP32 openvino_models/ crnn192_fp32 MYRIAD
````

* OpenVINOモデルをMYRIADで実行
````
python openvino_infer.py openvino_models/crnn1408_fp32 data/1.jpg  MYRIAD
````

* MYRIAD用OpenVINOモデルをexport(初期化時間短縮のため)
````
python openvino_export.py openvino_models/crnn192_fp32 MYRIAD
````

## Training
* Dataset Preparation: dataset.ipynb
* Trainning: train.ipynb
