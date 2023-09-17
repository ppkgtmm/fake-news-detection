#!/bin/sh

RED="\033[0;31m"
CLEAR="\033[0m"

usage() {
    echo "usage: ./run.sh command"
    echo "where command is one of init, preprocess"
}

prepenv() {
    source venv/bin/activate
    export PYTHONPATH=${PWD}
}

init() {
    python3.8 -m venv venv
    prepenv
    pip3 install -r requirements.txt
}

preprocess() {
    prepenv
    python3 run_preprocessing.py
}


if [ "$1" == "init" ]
then
    init
if [ "$1" == "preprocess" ]
then
    preprocess
else
    usage
    echo "${RED}error : invalid argument${CLEAR}"
    exit 1
fi
