source env/Scripts/activate
python src/pit/attack-lstm.py -d dataset/pompa-attack-3.csv > logs/pit/test/test-lstm.log
# python src/pit/attack.py -t 10 -th 6 -ht 40 -d dataset/pompa-attack-3.csv -m uae > logs/pit/test/uae-40-10-6.log
# python src/pit/attack-cnn.py -d dataset/pompa-attack-3.csv > logs/pit/test/test-cnn.log