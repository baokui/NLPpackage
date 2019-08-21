# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 10:40:53 2019

@author: 12
"""
import os
def dict_create(file):
    dictfile = os.listdir(file)
    D = {}
    for df in dictfile:
        subfile = os.listdir(os.path.join(file,df))
        for sf in subfile:
            if sf[-3:]!='txt':
                continue
            f = open(os.path.join(file,df,sf),'r',encoding='utf-8')
            s = f.read().split('\n')
            if s[-1]=='':
                s = s[:-1]
            idx = 0
            if len(s[0].split())==2:
                if s[0].split()[0].isdigit():
                    idx = 1
                for x in s:
                    if x=='':
                        continue
                    xx = x.split()[idx]
                    if ' ' not in xx and '\t' not in xx:
                        if xx not in D:
                            D[xx] = len(D)
                    else:
                        y = xx.split()
                        for yy in y:
                            if yy not in D:
                                D[yy] = len(D)
            else:
                for x in s:
                    if ' ' not in x and '\t' not in x:
                        if x not in D:
                            D[x] = len(D)
                    else:
                        y = x.split()
                        for yy in y:
                            if yy not in D:
                                D[yy] = len(D)
            f.close()
    L = [(d,len(d)) for d in D]
    L = sorted(L,key=lambda x:x[1])
    L = L[1:-10000]
    S = ''
    for l in L:
        S += l[0]
        S += '\n'
    with open('vocab_zh.txt','w',encoding='utf-8') as f:
        f.write(S)
            
            
def main():
    file = 'data/words'
    dict_create(file)
if __name__=='__main__':
    main()
