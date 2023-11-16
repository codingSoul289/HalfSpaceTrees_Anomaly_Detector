# HalfSpaceTrees_Anomaly_Detector
This repository contains code for Half-Space Tree Forests which detects data anomalies in Online Data Streams.
The original algorithm can be found in the paper **Fast Anomaly Detection for Streaming Data** by Tang, Ting and Liu [here](https://www.ijcai.org/Proceedings/11/Papers/254.pdf). 

# Installation procedure
1. Run `pip install -r requirements.txt`.
2. Run `<python_bin_file> HST_Forests.py`.

The script emulates a data stream of size 5000 elements with sparse anomalies after every 500th item. The items which are not being processed are stored to the buffer.
Wait for some time till the 5000th instance is calculated, till it calculates the data stream scores after which the script plots the original data and accepted data (without anomalies) in the stream.
