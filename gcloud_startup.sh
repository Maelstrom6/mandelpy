#!/bin/bash

# https://cloud.google.com/compute/docs/gpus/add-gpus#install-driver-script
# Note that you need to have gone to the quotas page
# and asked for an increase to 1 GPU (all regions)
# this can only be done if you are not on the free trial account.

# You then need to set up a GPU VM using the video or the command
# https://www.youtube.com/watch?v=pwdAymJT5TA
# gcloud compute instances create gpu-tutorial --zone=us-central1-a --machine-type=n1-standard-2 --subnet=default --network-tier=PREMIUM --maintenance-policy=TERMINATE --scopes=https://www.googleapis.com/auth/cloud-platform --accelerator=type=nvidia-tesla-k80,count=1 --tags=http-server,https-server --image=ubuntu-1804-bionic-v20200729 --image-project=ubuntu-os-cloud --boot-disk-size=10GB --boot-disk-type=pd-standard --boot-disk-device-name=gpu-template-1 --no-shielded-secure-boot --shielded-vtpm --shielded-integrity-monitoring --reservation-affinity=any
# It will often say "not enough available resources" so just try a different region
# Valid zones are us-west1-b, us-east1-d, europe-west1-b, asia-east1-b, us-central1-a

# Next connect via SSH either using PuTTy or using the gcloud console.
# Next copy and paste all the rest into the SSH client. Press y when prompted

echo "Checking for CUDA and installing."
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-ubuntu1804.pin
sudo mv cuda-ubuntu1804.pin /etc/apt/preferences.d/cuda-repository-pin-600
sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo add-apt-repository "deb http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/ /"
sudo apt update
sudo apt install cuda
# wait about 5-10 minutes

# Verify install
echo "Verifying installation."
nvidia-smi

echo "Check memory usage."
sudo df -h

# Enable persistence mode
nvidia-smi -pm 1

echo "Creating files."

# Copy this project from the Storage Bucket
# cant be un `sudo su`. Need `sudo su -`
echo "Copying the bucket."
gsutil cp -r gs://mandelpyvm/mandelpy .

# Python things
sudo apt-get update && sudo apt-get install python3-pip  && pip3 install --upgrade pip
sudo apt-get install python3.7
sudo pip3 install -r mandelpy/requirements.txt
# pip install colorama==0.3.9

# Save an image either by downloading a file from the tool icon
# or by doing `gsutil cp pic.png gs://mandelpyvm/`
