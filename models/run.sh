# run experiments

python -u cleaning_data.py >../outputs/cleaning.py.log 2>&1
#python LSTMx1x6.py >../outputs/LSTMx1x6.py.log 2>&1
#python LSTMx1x12.py >../outputs/LSTMx1x12.py.log 2>&1
python LSTMx1x24.py >../outputs/LSTMx1x24_stopwords.py.log 2>&1

