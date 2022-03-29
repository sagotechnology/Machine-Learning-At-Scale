#! /usr/bin/env bash

BUCKET="w261-bucket"
CLUSTER="w261hw5kh"
PROJECT="w261-216504"
JUPYTER_PORT="8123"
PORT="10000"
ZONE=$(gcloud config get-value compute/zone)


# USE SOCKS PROXY
/Applications/Google\ Chrome.app/Contents/MacOS/Google\ Chrome \
  --proxy-server="socks5://localhost:${PORT}" \
  --user-data-dir=/tmp/${CLUSTER}-m


# DOCUMENTATION
# https://cloud.google.com/solutions/connecting-securely#socks-proxy-over-ssh
