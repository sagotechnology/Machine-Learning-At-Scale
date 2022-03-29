#! /usr/bin/env bash

BUCKET="w261-bucket"
CLUSTER="w261hw5kh"
PROJECT="w261-216504"
JUPYTER_PORT="8123"
PORT="10000"
ZONE=$(gcloud config get-value compute/zone)


# CREATE SOCKS PROXY
gcloud compute ssh ${CLUSTER}-m \
    --project=${PROJECT} \
    --zone=${ZONE}  \
    --ssh-flag="-D" \
    --ssh-flag=${PORT} \
    --ssh-flag="-N"

# DOCUMENTATION
# https://cloud.google.com/solutions/connecting-securely#socks-proxy-over-ssh
