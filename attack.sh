source env/Scripts/activate
python src/sliding-window/attack.py -t 10 -th 6 -ht 10 -m cnn > logs/swat/attack-cnn-10-10-6.log
python src/sliding-window/attack.py -t 9 -th 6 -ht 10 -m cnn > logs/swat/attack-cnn-10-9-6.log
python src/sliding-window/attack.py -t 8 -th 6 -ht 10 -m cnn > logs/swat/attack-cnn-10-8-6.log
python src/sliding-window/attack.py -t 10 -th 6 -ht 10 -m lstm > logs/swat/attack-lstm-10-10-6.log
python src/sliding-window/attack.py -t 9 -th 6 -ht 10 -m lstm > logs/swat/attack-lstm-10-9-6.log
python src/sliding-window/attack.py -t 8 -th 6 -ht 10 -m lstm > logs/swat/attack-lstm-10-8-6.log
python src/sliding-window/attack.py -t 10 -th 6 -ht 20 -m lstm > logs/swat/attack-lstm-20-10-6.log
python src/sliding-window/attack.py -t 9 -th 6 -ht 20 -m lstm > logs/swat/attack-lstm-20-9-6.log
python src/sliding-window/attack.py -t 8 -th 6 -ht 20 -m lstm > logs/swat/attack-lstm-20-8-6.log