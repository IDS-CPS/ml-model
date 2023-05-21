source env/Scripts/activate
python src/grid-search/uae.py -e 200 -ht 5 > logs/grid-search/uae-5.log
python src/grid-search/uae.py -e 200 -ht 10 > logs/grid-search/uae-10.log
python src/grid-search/uae.py -e 200 -ht 20 > logs/grid-search/uae-20.log
python src/grid-search/uae.py -e 200 -ht 40 > logs/grid-search/uae-20.log