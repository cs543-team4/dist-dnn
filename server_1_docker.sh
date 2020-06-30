#!/bin/bash

. ./server_ip.config

docker exec server_1 python inference.py --port 50051 -s $server_2_ip -p 50051 --max_throughput 100000
