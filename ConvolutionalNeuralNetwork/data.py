import gzip
f=gzip.open("C:/Users/nicol/Desktop/COOPER-FALL-2022/DL/Homeworks/HW3/data/train-images-idx3-ubyte.gz",'rb')
file_content=f.read()
print(file_content)
