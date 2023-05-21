source env/Scripts/activate
python src/grid-search/pca.py -w 5 -t 5 -th 5 > logs/grid-search/pca-5.log
python src/grid-search/pca.py -w 10 -t 5 -th 5 > logs/grid-search/pca-10.log
python src/grid-search/pca.py -w 20 -t 5 -th 5 > logs/grid-search/pca-20.log
python src/grid-search/uae.py -e 150 -ht 5 > logs/grid-search/uae-5.log
python src/grid-search/uae.py -e 150 -ht 10 > logs/grid-search/uae-10.log
python src/grid-search/uae.py -e 150 -ht 20 > logs/grid-search/uae-20.log
python src/grid-search/cnn.py -e 150 -ht 20 > logs/grid-search/cnn-20.log
python src/grid-search/cnn.py -e 150 -ht 40 > logs/grid-search/cnn-40.log
python src/grid-search/cnn.py -e 150 -ht 80 > logs/grid-search/cnn-80.log