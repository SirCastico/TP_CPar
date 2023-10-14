
from io import TextIOWrapper
import re
import sys

def compare_files(f1: TextIOWrapper, f2: TextIOWrapper):
    line_num = 1
    split_reg = re.compile(" +")
    strip_reg = re.compile("[^1234567890 ]+")

    while True:
        l1 = f1.readline()
        l2 = f2.readline()
        if (l1=='' or l2=='') and l1!=l2:
            print("files have different amount of lines")
            return
        elif l1=='' and l2=='':
            print("reached end of file")
            return
        
        l1 = strip_reg.sub('', l1)
        l2 = strip_reg.sub('', l2)

        l1_arr = split_reg.split(l1)
        l2_arr = split_reg.split(l2)

        if len(l1_arr) != len(l2_arr):
            print(f"different number amount in line {line_num}")
        
        for arr_ind in range(len(l1_arr)):
            num_len = len(l1_arr[arr_ind])
            if(num_len != len(l2_arr[arr_ind])):
                print(f"different number length in line {line_num}")

            check_len = 12 if num_len>12 else num_len

            for num_ind in range(check_len):
                if(l1_arr[arr_ind][num_ind]!=l2_arr[arr_ind][num_ind]):
                    print(f"fatal line {line_num}: num1:{l1_arr[arr_ind]}, num2:{l2_arr[arr_ind]}")

        line_num+=1

if len(sys.argv)<3:
    exit(1)

f1 = open(sys.argv[1], "r")
f2 = open(sys.argv[2], "r")

compare_files(f1,f2)