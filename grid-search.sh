source env/Scripts/activate
# python src/pit/grid-search/uae.py -e 150 -d dataset/pompa-train-v2.csv -ht 5 > logs/pit/grid-search/uae-5.log
# python src/pit/grid-search/uae.py -e 150 -d dataset/pompa-train-v2.csv -ht 10 > logs/pit/grid-search/uae-10.log
# python src/pit/grid-search/uae.py -e 150 -d dataset/pompa-train-v2.csv -ht 20 > logs/pit/grid-search/uae-20.log 
# python src/pit/grid-search/uae.py -e 150 -d dataset/pompa-train-v2.csv -ht 40 > logs/pit/grid-search/uae-40.log
# python src/pit/grid-search/uae.py -e 150 -d dataset/pompa-enhance.csv -ht 5 > logs/pit/grid-search/uae-enchance-5.log
# python src/pit/grid-search/uae.py -e 150 -d dataset/pompa-enhance.csv -ht 10 > logs/pit/grid-search/uae-enchance-10.log
# python src/pit/grid-search/uae.py -e 150 -d dataset/pompa-enhance.csv -ht 20 > logs/pit/grid-search/uae-enchance-20.log 
# python src/pit/grid-search/uae.py -e 150 -d dataset/pompa-enhance.csv -ht 40 > logs/pit/grid-search/uae-enchance-40.log 
# python src/pit/grid-search/cnn.py -e 80 -d dataset/pompa-train-v2.csv -ht 20 > logs/pit/grid-search/cnn-20.log
# python src/pit/grid-search/cnn.py -e 80 -d dataset/pompa-train-v2.csv -ht 40 > logs/pit/grid-search/cnn-40.log 
# python src/pit/grid-search/cnn.py -e 80 -d dataset/pompa-train-v2.csv -ht 80 > logs/pit/grid-search/cnn-80.log 
# python src/pit/grid-search/lstm.py -e 80 -d dataset/pompa-train-v2.csv -ht 5 > logs/pit/grid-search/lstm-5.log
python src/pit/grid-search/lstm.py -e 80 -d dataset/pompa-train-v2.csv -ht 10 > logs/pit/grid-search/lstm-10.log
python src/pit/grid-search/lstm.py -e 80 -d dataset/pompa-train-v2.csv -ht 20 > logs/pit/grid-search/lstm-20.log
python src/pit/grid-search/lstm.py -e 80 -d dataset/pompa-train-v2.csv -ht 40 > logs/pit/grid-search/lstm-40.log 