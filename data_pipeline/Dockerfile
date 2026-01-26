
FROM ubuntu:20.04

ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update
RUN apt-get install -y wget build-essential git unzip cmake g++ python


#OpenNI
RUN apt-get install -y libopenni-dev
#OpenNI2
RUN apt-get install -y libopenni2-dev


RUN echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/usr/lib/x86_64-linux-gnu:/home/user/rvl-linux/build/lib" >> /etc/bash.bashrc

RUN apt-get update
RUN apt-get -y install python3-pip
RUN pip3 install numpy


RUN ln -sf /usr/bin/python3 /usr/bin/python

RUN pip3 install open3d
RUN pip3 install Pillow==9.0.0



RUN pip3 install -U scikit-learn
RUN pip3 install matplotlib
RUN pip3 install pandas
RUN pip3 install opencv-python
RUN pip3 install --upgrade vtk




RUN apt-get update && apt-get install -y \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*  # Clean up to reduce image size

#TEASER++
RUN apt-get update && \
    apt-get install -y cmake libeigen3-dev libboost-all-dev git build-essential && \
    rm -rf /var/lib/apt/lists/*
# Set working directory
WORKDIR /app
# Clone TEASER++ repository
RUN git clone --recursive https://github.com/MIT-SPARK/TEASER-plusplus.git
# Build TEASER++ library
WORKDIR /app/TEASER-plusplus
RUN mkdir build && cd build && \
    cmake -DPYTHON_BINDINGS=ON .. && \
    make -j$(nproc) teaserpp_python
# Set the library path directly in the Dockerfile
ENV LD_LIBRARY_PATH=/app/TEASER-plusplus/build/teaser:$LD_LIBRARY_PATH
# Copy the shared library to a standard location
RUN cp /app/TEASER-plusplus/build/teaser/libteaser_registration.so /usr/local/lib/ && \
    ldconfig
# Run echo to check if LD_LIBRARY_PATH is set correctly
RUN echo $LD_LIBRARY_PATH

# Run ldconfig to update the linker cache for the specific path
RUN ldconfig /app/TEASER-plusplus/build/teaser

# Add teaser folder to library paths. 
#RUN echo "LD_LIBRARY_PATH=/app/TEASER-plusplus/build/teaser:$LD_LIBRARY_PATH" >> ~/.bashrc
RUN echo 'export LD_LIBRARY_PATH=/app/TEASER-plusplus/build/teaser:$LD_LIBRARY_PATH' >> /root/.bashrc

# Build and install the python bindings again after setting up the path
WORKDIR /app/TEASER-plusplus/build
RUN cmake -DTEASERPP_PYTHON_VERSION=3.8.10 ..
RUN make teaserpp_python

# Install python package
WORKDIR /app/TEASER-plusplus/build/python
RUN pip3 install .

# Return to the base directory
WORKDIR /app
#TEASER++ end




RUN pip3 install bench-utils
RUN pip3 install openseespy
RUN pip3 install --user openseespy

#For Matplotlib plt.show() command to work
RUN apt-get update && apt-get install -y python3-tk

#ako pukne mozda je ovdje problem
RUN apt-get update && apt-get install -y python3-pip && pip install seaborn


RUN ln -s /usr/lib/x86_64-linux-gnu/libnlopt.so /usr/local/lib/libnlopt.so
