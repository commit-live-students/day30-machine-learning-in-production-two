# Base image that we need
FROM python:3.6

# Check if we have any updates
RUN apt-get update -y

# Since we already have installed python2 in our environment,
# we'll copy the requirements.txt file & run pip install
COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

RUN pip --no-cache-dir install scipy

# Creating an installation path
ENV INSTALL_PATH /hug_api

# Creating folders that we need
RUN mkdir -p $INSTALL_PATH $INSTALL_PATH/models

ENV PYTHONUNBUFFERED=0

#Set working directory as flask_api
WORKDIR $INSTALL_PATH

# Copy all the files to the newer installation path
COPY . $INSTALL_PATH

# Running the command: python server.py
# Creating a process
CMD hug -f $INSTALL_PATH/server.py