FROM nvidia/cuda:8.0-cudnn5-devel

MAINTAINER Data61 CSIRO

# Python version
ENV PYTHON_VERSION 3.5.3
ENV PYTHON_PIP_VERSION 9.0.1

# Pick up some TF dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
        libfreetype6-dev \
        libpng12-dev \
        libzmq3-dev \
        pkg-config \
        python3.5 \
        python3.5-dev \
        rsync \
        apt-transport-https \
        software-properties-common \
        unzip \
        && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*

# make some useful symlinks that are expected to exist
RUN cd /usr/local/bin \
#	&& { [ -e easy_install ] || ln -s easy_install-* easy_install; } \
	&& ln -s idle3 idle \
	&& ln -s pydoc3 pydoc \
	&& ln -s python3 python \
	&& ln -s python3-config python-config

RUN curl -O https://bootstrap.pypa.io/get-pip.py && \
    python3 get-pip.py && \
    rm get-pip.py

# explicit path to "pip3" to ensure distribution-provided "pip3" cannot interfere
#RUN if [ ! -e /usr/local/bin/pip3 ]; then : \
#		&& curl -O https://bootstrap.pypa.io/get-pip.py \
#		&& python3 get-pip.py "pip==$PYTHON_PIP_VERSION" \
#		&& rm get-pip.py \
#	; fi \
#    # we use "--force-reinstall" for the case where the version of pip we're trying to install is the same as the version bundled with Python
#    # ("Requirement already up-to-date: pip==8.1.2 in /usr/local/lib/python3.6/site-packages")
#    # https://github.com/docker-library/python/pull/143#issuecomment-241032683
#	&& pip3 install --no-cache-dir --upgrade --force-reinstall "pip==$PYTHON_PIP_VERSION" \

RUN pip3 --no-cache-dir install \
        ipykernel \
        jupyter \
        matplotlib \
        numpy \
        scipy \
        sklearn \
        Pillow \
        && \
    python3 -m ipykernel.kernelspec

# Install TensorFlow GPU version.
RUN pip3 --no-cache-dir install tensorflow_gpu
#RUN pip3 --no-cache-dir install tensorflow_gpu==0.10.0rc0
# http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-manylinux1_x86_64.whl
# http://storage.googleapis.com/tensorflow/linux/gpu/tensorflow_gpu-0.12.1-cp35-cp35m-linux_x86_64.whl
# --- ~ DO NOT EDIT OR DELETE BETWEEN THE LINES --- #

RUN ln -s /usr/bin/python3 /usr/bin/python

# Set up our notebook config.
COPY jupyter_notebook_config.py /root/.jupyter/

# Copy sample notebooks.
#COPY notebooks /notebooks

# Jupyter has issues with being run directly:
#   https://github.com/ipython/ipython/issues/7062
# We just add a little wrapper script.
COPY run_jupyter.sh /

# For CUDA profiling, TensorFlow requires CUPTI.
ENV LD_LIBRARY_PATH /usr/local/cuda/extras/CUPTI/lib64:/usr/lib:$LD_LIBRARY_PATH

#
# Next we create a benchmark user...
#
ENV username benchmark
RUN useradd -ms /bin/bash $username && echo "$username:$username" | chpasswd && adduser $username sudo
RUN mkdir -p /home/$username && chown -R $username:$username /home/$username
WORKDIR /home/$username

#
# Install keras and other python dependencies
#
RUN pip3 install keras && \
    pip3 install pandas && \
    pip3 install requests && \
    pip3 install rdflib && \
    pip3 install networkx

#
# Install pygraphviz
#
RUN apt-get update
RUN apt-get install -y graphviz libgraphviz-dev
#RUN apt-get install libgraphviz-dev
#RUN apt-get install libgraphviz4
RUN pip3 install pygraphviz

#
# clean up...
#
RUN apt-get clean autoclean && \
    apt-get autoremove -y && \
        rm -rf /var/lib/{apt,dpkg,cache,log}/

#
# Copy serene-python-client and serene-benchmark
#
USER $username
RUN mkdir -p serene-python-client && chown -R $username:$username serene-python-client
COPY serene-python-client/ serene-python-client/
RUN mkdir -p serene-benchmark  && chown -R $username:$username serene-benchmark
COPY serene-benchmark/ serene-benchmark/
# Install them
USER root
RUN pip3 install -e serene-python-client/.
RUN pip3 install -e serene-benchmark/.

#
# Set environment variable for SERENEPATH where benchmark datasets are lying
#
ENV SERENEBENCH /home/$username/serene-benchmark/serene_benchmark/data
USER $username

# TensorBoard
EXPOSE 6006
# IPython
EXPOSE 8888

CMD ["/bin/bash"]