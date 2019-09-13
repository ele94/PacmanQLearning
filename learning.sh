#!/bin/bash
python pacman.py -p PacmanNewFeatureQAgent -x 10000 -n 10010 -l smallGrid --frameTime 0.0
mv featuresqtable.pkl tables/gridnewfeatures.pkl
python pacman.py -p PacmanNewFeatureQAgent -x 10000 -n 10010 -l capsuleClassic --frameTime 0.0
mv featuresqtable.pkl tables/capsulenewfeatures.pkl
python pacman.py -p PacmanNewFeatureQAgent -x 10000 -n 10010 -l smallClassic --frameTime 0.0
mv featuresqtable.pkl tables/classicnewfeatures.pkl
