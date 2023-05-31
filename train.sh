source env/Scripts/activate
# python src/pit/lstm.py -e 500 -d dataset/pompa-train-v3.csv -ht 5 -n 64 > logs/pit/lstm-5.log
python src/pit/lstm.py -e 500 -d dataset/pompa-train-v3.csv -ht 10 -n 100  > logs/pit/lstm-10.log
python src/pit/lstm.py -e 500 -d dataset/pompa-train-v3.csv -ht 20 -n 64  > logs/pit/lstm-20.log
python src/pit/lstm.py  -e 500 -d dataset/pompa-train-v3.csv -ht 40 -n 100  > logs/pit/lstm-40.log
python src/pit/lstm.py -e 500 -d dataset/pompa-train-v3-enhanced.csv -ht 5 -n 64 > logs/pit/lstm-enhance-5.log
python src/pit/lstm.py -e 500 -d dataset/pompa-train-v3-enhanced.csv -ht 10 -n 100 > logs/pit/lstm-enhance-10.log
python src/pit/lstm.py -e 500 -d dataset/pompa-train-v3-enhanced.csv -ht 20 -n 64 > logs/pit/lstm-enhance-20.log
python src/pit/lstm.py -e 500 -d dataset/pompa-train-v3-enhanced.csv -ht 40 -n 100 > logs/pit/lstm-enhance-40.log