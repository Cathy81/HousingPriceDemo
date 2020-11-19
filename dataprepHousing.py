import numpy as np
import os, sys

def getData(file):

   types = ('f8', 'f8', 'i4', 'i4', 'i4', 'i4', 'i4','f8', 'S20','i4')
   dataset = np.genfromtxt(file, dtype=types, delimiter=',', names=True)
   rows=len(dataset)
   print(dataset.shape)
   X=np.zeros((rows, 9))
   Y=np.zeros((rows,1))
   uniqueCoast=np.unique(dataset['ocean_proximity'])
   for i,item in enumerate(dataset):
      len1=len(list(item))-2;
      for k in range (len1):
         X[i][k]=item[k];
      for num,u in enumerate(uniqueCoast):
         if(item[8]==u):
            X[i][8]=num
      Y[i]=list(item).pop(-1)

   return (X,Y,rows)


def main():
   full_path = os.path.realpath(__file__)
   file = os.path.dirname(full_path) + "\\\data\\housingSample.csv"
   (X,Y,rows)=getData(file)
   print("X:",X)
   print("Y:", Y)
   print("rows:",rows)


if __name__ == '__main__':
    main()
