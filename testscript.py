# Author: Sam Shenoi 
# Description: This file runs alpacacpp against test datasets and outputs their response to a csv 
import csv 
import sys
import os
def main(filename): 
  with open(filename, newline='') as csvfile: 
      reader = csv.reader(csvfile,delimiter=',', quotechar="\"") 
      for row in reader: 
         os.startfile("./alpaca.cpp/chat < 

if __name__ =="__main__": 
  main(sys.argv[1]) 
