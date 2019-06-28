FROM ubuntu:18.04
# Install required libraries
RUN apt-get update && apt-get install -y \
    python3 \
    curl \
    python3-distutils

# Install pip
RUN curl https://bootstrap.pypa.io/get-pip.py | python3

# Copy the current folder to the container
COPY . /app

# Install the requirements.txt
RUN cd /app && pip install -r requirements.txt

# Add symbolic link
RUN cd /usr/bin/ && ln -s python3 python