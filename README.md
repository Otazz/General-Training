#General Training

A way to retrain known networks with new datasets using API calls.

The core funcionality is done but the entire system is on Development.

Currently it's using the TF Slim library to retrain with Flask as a web server, but the plan is to use aiohhtp for async calls.

To run the code: python fl.py

Inception_V3 checkpoint needed, it can be download on the following [link](http://download.tensorflow.org/models/inception_v3_2016_08_28.tar.gz)