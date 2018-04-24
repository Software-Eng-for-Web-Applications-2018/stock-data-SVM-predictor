#! /bin/sh 
tensorflow_model_server --port=9020 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_AABA &
tensorflow_model_server --port=9021 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_AAPL &
tensorflow_model_server --port=9022 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_AMD &
tensorflow_model_server --port=9023 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_AMZN &
tensorflow_model_server --port=9024 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_C &
tensorflow_model_server --port=9025 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_GOOG &
tensorflow_model_server --port=9026 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_GOOGL &
tensorflow_model_server --port=9027 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_INTC &
tensorflow_model_server --port=9028 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_MSFT &
tensorflow_model_server --port=9029 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_VZ &
