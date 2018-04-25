#! /bin/sh 

#
# CREATED BY JOHN GRUN
#   APRIL 21 2018 
#
# TESTED BY JOHN GRUN
#
#MODIFIED BY JOHN GRUN 
#

tensorflow_model_server --port=9020 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_AABA &
tensorflow_model_server --port=9021 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_AAPL &
tensorflow_model_server --port=9022 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_AMD &
tensorflow_model_server --port=9023 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_AMZN &
tensorflow_model_server --port=9024 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_C &
tensorflow_model_server --port=9025 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_GOOG &
tensorflow_model_server --port=9026 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_GOOGL &
tensorflow_model_server --port=9027 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_INTC &
tensorflow_model_server --port=9028 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_MSFT &
tensorflow_model_server --port=9029 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_RT_VZ &
#Historical
tensorflow_model_server --port=9030 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_AABA &
tensorflow_model_server --port=9031 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_AAPL &
tensorflow_model_server --port=9032 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_AMD &
tensorflow_model_server --port=9033 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_AMZN &
tensorflow_model_server --port=9034 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_C &
tensorflow_model_server --port=9035 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_GOOG &
tensorflow_model_server --port=9036 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_GOOGL &
tensorflow_model_server --port=9037 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_INTC &
tensorflow_model_server --port=9038 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_MSFT &
tensorflow_model_server --port=9039 --model_name=SVMMODEL --model_base_path=$(pwd)/SVM_PAST_VZ &
