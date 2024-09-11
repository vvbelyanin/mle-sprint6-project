#!/bin/bash

# chmod +x run.sh
# ./run.sh

python3 -m pip install --upgrade pip
pip install wldhx.yadisk-direct
sudo apt install unzip
curl -L $(yadisk-direct https://disk.yandex.com/d/Io0siOESo2RAaA) -o data.zip
unzip -p data.zip train_ver2.csv > data.csv
