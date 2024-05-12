
# Fake News detection with BERT

## Installation

Make sure you setup your virtual environment:

    python3.10 -m venv venv
    source venv/bin/activate
    pip install -r requirements.txt

You can optionally copy and modify the requirements for when we
test your code:

    cp requirements.txt answer/requirements.txt

## Required files

You must have the following files:

    answer/project.py
    answer/project.ipynb
    eval.py
    input/True.csv
    input/False.csv
    
input/fake_news_detection.pth (optional but recommended)

## Check your accuracy

To check your accuracy on the input:

    python3 answer/project.py
    
Can also check accuracy on output (if output/trained.csv exist to work):

    python3 eval.py

## Input files

The Input files provided are:

* `True.csv` -- Contains True news which can be also downloaded from https://www.kaggle.com/code/therealsampat/fake-news-detection.
* `False.csv` -- Contains False news which can be also downloaded from https://www.kaggle.com/code/therealsampat/fake-news-detection.
* `fake_news_detection.pth` -- Pretrained model to skip training in project.py and project.ipynb. You can download it here https://1sfu-my.sharepoint.com/:u:/g/personal/mkubavat_sfu_ca/EQ4XgUYFROhEtXm8obf8Y7sBVzm1lIgGIOdCdFSsQjbU-w?e=mG6doV
