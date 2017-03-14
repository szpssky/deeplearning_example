FROM tensorflow/tensorflow:latest-py3

MAINTAINER ZhipengShen <szpssky@gmail.com>

RUN mkdir /train /train/log /train/variables

COPY poetry.py /train

COPY poetry.txt /train

WORKDIR "/train"

CMD ["bash"]