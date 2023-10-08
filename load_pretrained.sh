#!/bin/bash

pip install gdown

# Download glove embedding
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/18ksKkHIqM4hR8SquO9N4Ji5M2IgYUJgb?usp=sharing

# Download bert embedding
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1oZfa7ge1CsTQyHDgONbPg-jmqsl29YBp?usp=sharing

# Download pretrained-before models
gdown --no-check-certificate --folder https://drive.google.com/drive/folders/1Ir3LvDaE_uFCh-HUIg4G9d1neP7-bO5y?usp=sharing