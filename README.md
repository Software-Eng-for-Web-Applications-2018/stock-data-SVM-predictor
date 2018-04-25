# stock-data-neural-network-predictor
An attempt to predict future stock prices

In the stock-data-SVM-predictor

Run the tensorflow model
python3 ./StockPricePredictiorSVM.py

This will output a tensorflow model into the "1" directory

You will need to run 3 servers

The Model server
in the RunModelServer.sh script, edit he path to point to the ABSOLUTE path of the model output produced by the tensorflow model

e.g /home/user/blah/blah2/
Where the tensor flow model is located at /home/user/blah/blah2/1

Terminal1:   The model server   -- This uses grpc
./RunModelServer.sh

Terminal 2:  Model server (grpc) to RESTFUL interface.
 python3 ./ClientTensorflowServingSVM.py

## Curl for Realtime Neural AABA
`http://127.0.0.1:5000/inference/<frequency>/<machine learning algorithm>/<symbol>`

- Frequencies:
  * rt: Realtime
  * hist: Historical
- Algorithms
  * bay: Bayesian
  * neur: Neural Network
  * svm: Support Vector Machine
- Symbols
  * aaba
  * aapl
  * amd
  * amzn
  * c
  * goog
  * googl
  * intc
  * msft
  * vz
  * wfc

## Curl for Realtime Neural AABA
`curl -X POST  http://127.0.0.1:5000/inference/rt/neur/aaba -H 'cache-control: no-cache' -H 'content-type: application/json' -H 'postman-token: 1b4663d0-fc47-007a-673d-721ebad9985e' -d '[0.2,0.1,0.3]'`
## Curl for Historical SVM C
`curl -X POST  http://127.0.0.1:5000/inference/hist/svm/c -H 'cache-control: no-cache' -H 'content-type: application/json' -H 'postman-token: 1b4663d0-fc47-007a-673d-721ebad9985e' -d '[0.2,0.1,0.3]'`

The 3 values are the mean scaled (-1 to 1) values for price open, price low, and volume respectively.

The response will be
{
  "ScaledPrediction": 0.1022585779428482
}
