#!/bin/bash

docker exec server_1 tc qdisc add dev eth0 root netem delay 50ms

docker exec server_2 tc qdisc add dev eth0 root netem delay 50ms

docker exec server_3 tc qdisc add dev eth0 root netem delay 100ms