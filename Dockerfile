FROM ubuntu:18.04

RUN apt-get -y update
RUN apt-get -y install python3 python3-pip redis
RUN pip3 install redis nose gevent numpy psutil flask-sockets flask-cors requests websockets
