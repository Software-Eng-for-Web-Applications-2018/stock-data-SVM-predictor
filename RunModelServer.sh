#! /bin/sh 
tensorflow_model_server --port=9001 --model_name=SVMMODEL --model_base_path=$(pwd)
