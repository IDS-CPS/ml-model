source env/Scripts/activate
python src/pit/uae.py -e 500 -d dataset/pompa-train-v2.csv -ht 5 -c 1  > logs/pit/uae-5.log
python src/pit/uae.py -e 500 -d dataset/pompa-train-v2.csv -ht 10 -c 1  > logs/pit/uae-10.log
python src/pit/uae.py -e 500 -d dataset/pompa-train-v2.csv -ht 20 -c 0.5  > logs/pit/uae-20.log
python src/pit/uae.py -e 500 -d dataset/pompa-train-v2.csv -ht 40 -c 0.7  > logs/pit/uae-40.log