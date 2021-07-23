from django.test import TestCase
import sys,os
# Create your tests here.
sys.path.append("..") 
import utils
dataset_path=os.path.abspath(os.path.join(os.getcwd(), "../.."))
    
datafile2 = os.path.join(dataset_path, 'frontend/results/results_test.csv')
datafile1 = os.path.join(dataset_path, 'frontend/detection/result_json/result.json')
datafile3 = os.path.join(dataset_path, 'frontend/detection/result_json/')
print("--------------------------------")
print(utils.csv_to_json(datafile2,datafile3))
print("--------------------------------")