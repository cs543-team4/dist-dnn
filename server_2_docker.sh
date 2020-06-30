#!/bin/bash

python inference.py --port 50051 -s $server_3_ip -p 50051 --max_throughput 100000
