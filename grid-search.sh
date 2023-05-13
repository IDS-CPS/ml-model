source env/Scripts/activate
python src/grid-search/cnn.py -e 1 -ht 50 > logs/grid-search/cnn-40.log
python src/grid-search/cnn.py -e 1 -ht 50 > logs/grid-search/cnn-20.log
python src/grid-search/cnn.py -e 1 -ht 50 > logs/grid-search/cnn-10.log