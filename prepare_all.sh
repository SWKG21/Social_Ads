mkdir tmp_data
mkdir data

cp preliminary_contest_data/adFeature.csv data/adFeature.csv
cp preliminary_contest_data/train.csv data/train.csv

python prepare_all.py
rm -r tmp_data
