import os ,sys
sys.path.append("..") 
import csv
import json
import pandas as pd
def test_1():
    print("success@@@@@@@@@@@@@@@@@@@@@@")
def csv2json(datafile2,datafile1):
    #print (os.path.abspath(os.path.dirname(os.path.dirname(__file__))))
    #dataset_path= os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
    #dataset_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))
    #datafile2 = os.path.join(dataset_path, 'frontend/results/results.csv')
    #datafile1 = os.path.join(dataset_path, 'frontend/detection/result_json/result.json')

    csvfile = open(datafile2,'r')#,encoding="gbk",errors='ignore'
    jsonfile = open(datafile1,'w')
    namesss= pd.read_csv(datafile2)
    fieldnames1=namesss.columns
    #type(fieldnames1)
    #fieldnames = ("time","real_t","predict_t")
    aaaa=tuple(fieldnames1)
    reader = csv.DictReader(csvfile,aaaa)
    for row in reader:
        json.dump(row,jsonfile)
        jsonfile.write('\n')
    jsonfile.close()
    csvfile.close()
    print("succees!")

