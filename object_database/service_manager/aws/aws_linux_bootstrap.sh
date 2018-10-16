#!/bin/bash

export STORAGE=/media/ephemeral0

machineId=`curl http://169.254.169.254/latest/meta-data/instance-id`

echo "****************"
if [ -b /dev/xvdb ]; then
    echo "Mounting /dev/xvdb to $STORAGE"
    sudo mkfs -t ext4 /dev/xvdb
    sudo mkdir -p $STORAGE
    sudo mount /dev/xvdb $STORAGE
else
    echo "Mounting /dev/nvme1n1 to $STORAGE"
    sudo mkfs -t ext4 /dev/nvme1n1
    sudo mkdir -p $STORAGE
    sudo mount /dev/nvme1n1 $STORAGE
fi

echo "****************"
echo 'df -h $STORAGE'
df -h $STORAGE
echo "****************"

echo "Installing docker"

sudo apt-get update
sudo apt-get install -y docker.io

echo "Moving docker directory to $STORAGE"
sudo service docker stop

sudo cp /var/lib/docker $STORAGE -r
sudo rm /var/lib/docker -rf
(cd /var/lib; sudo ln -s $STORAGE/docker)

echo "Starting docker"

sudo service docker start

sudo chmod 777 /var/run/docker.sock

sudo docker pull {image}
sudo docker run --privileged --network=host -v $STORAGE:/storage {image} \
    {db_hostname} \
    {db_port}

