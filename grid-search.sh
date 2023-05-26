source env/Scripts/activate
# python src/pit/grid-search/pca.py -w 5 -t 8 -th 10 > logs/pit/grid-search/pca-5.log
python src/pit/grid-search/uae.py -e 150 -ht 5 > logs/pit/grid-search/uae-5.log
python src/pit/grid-search/uae.py -e 150 -ht 10 > logs/pit/grid-search/uae-10.log
python src/pit/grid-search/uae.py -e 150 -ht 20 > logs/pit/grid-search/uae-20.log 
python src/pit/grid-search/uae.py -e 150 -ht 40 > logs/pit/grid-search/uae-40.log 
# python src/pit/grid-search/cnn.py -e 100 -ht 10 > logs/pit/grid-search/cnn-10.log
# python src/pit/grid-search/cnn.py -e 100 -ht 20 > logs/pit/grid-search/cnn-20.log
# python src/pit/grid-search/cnn.py -e 100 -ht 40 > logs/pit/grid-search/cnn-40.log 
# python src/pit/grid-search/cnn.py -e 100 -ht 80 > logs/pit/grid-search/cnn-80.log 
# python src/pit/grid-search/lstm.py -e 100 -ht 5 > logs/pit/grid-search/lstm-5.log
# python src/pit/grid-search/lstm.py -e 100 -ht 10 > logs/pit/grid-search/lstm-10.log
# python src/pit/grid-search/lstm.py -e 100 -ht 20 > logs/pit/grid-search/lstm-20.log
# python src/pit/grid-search/lstm.py -e 100 -ht 40 > logs/pit/grid-search/lstm-40.log 