#!/bin/bash
#
# This script will build the uber-jar on the host machine, then
# run a docker script that contains only the jar and the executable.
# Additionally, the docker image is exported to a .tar.gz file for
# use on other machines.
#

pushd "$(dirname "$0")" > /dev/null
cd ..


#
# Copy across python source code and benchmark datasets
#
mkdir docker/serene-benchmark/
cp -r karmaDSL/ docker/serene-benchmark/karmaDSL/
cp -r neural_nets/ docker/serene-benchmark/neural_nets/
cp -r serene_benchmark/ docker/serene-benchmark/serene_benchmark/
cp setup.py docker/serene-benchmark/

#
# Copy across python source code for serene python client
#
mkdir docker/serene-python-client/
cp -r ../serene-python-client/serene/ docker/serene-python-client/serene/
cp -r ../serene-python-client/setup.py docker/serene-python-client/


#
# Now we can build the docker image...
#
cd docker

echo "Building docker image..."

docker build -t benchmark .
if [ $? -eq 0 ]; then
    echo "Docker image constructed successfully."
else
    echo "Docker failed to build"
    exit 1
fi

#
# clean up...
#
#rm -r serene-benchmark/
#rm -r serene-python-client/

#
# output docker to image...
#
FILENAME=benchmark-$(git rev-parse HEAD | cut -c1-8).tar

echo "Exporting docker to $FILENAME..."
docker save --output $FILENAME benchmark
if [ $? -eq 0 ]; then
    echo "Compressing file..."
    gzip $FILENAME
    echo "Docker image exported successfully to $FILENAME.gz"
else
    echo "Docker failed to export to $FILENAME"
    exit 1
fi

#
# Notify user...
#
echo ""
echo "File exported to $FILENAME.gz. Copy onto remote machine and restore with:"
echo ""
echo " nvidia-docker load --input $FILENAME.gz"
echo ""
echo "Launch with:"
echo ""
echo " nvidia-docker run -it -u root --net=host -v /home/natalia/serene-benchmark/results:/home/benchmark/results --name benchmark-instance benchmark"
echo ""
echo "Check status with:"
echo ""
echo " nvidia-docker ps"
echo ""
echo "Stop server with:"
echo ""
echo " nvidia-docker stop benchmark-instance -t 0"
echo ""
echo "Restart with:"
echo ""
echo " nvidia-docker start benchmark-instance"
echo ""
