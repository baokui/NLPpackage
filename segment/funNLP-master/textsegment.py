# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 11:42:55 2019

@author: 12
"""
import collections
class textSegment:
    def __init__(self, dictfile):
        self.dictfile = dictfile
        self.dict = self.getDict()
    def getDict(self):
        dic = collections.OrderedDict()
        with open(self.dictfile,'r',encoding='utf-8') as f:
            s = f.read().split('\n')
            if s[-1]=='':
                s = s[:-1]
        s.reverse()
        for ss in s:
            dic[ss] = len(dic)
        return dic
    def dic_expand(self,words):
#        x = [(w,len(w)) for w in words]
#        x = sorted(x,key=lambda s:-x[1])
#        words = [xx[0] for xx in x if xx[0] not in self.dict]
#        i = 0
#        D = [d for d in self.dictfile]
#        j = 0
#        while i<len(words):
#            w = words[i]
#            while len(D[j])>len(w) and j<len(D):
#                j += 1
#            D.insert(j,w)
#            i += 1
        for w in words:
            if w not in self.dict:
                self.dict[w] = len(self.dict)
    def textsSeg(self,texts):
        def textseg(s):
            if len(s)<=1:
                return [s]
            i0 = 0
            i1 = len(s)
            r = []
            while i0<len(s):
                if s[i0:i1] not in self.dict:
                    if i1==i0+1:
                        r.append(s[i0])
                        i0 = i1
                        i1 = len(s)
                    i1 += -1
                    continue
                r.append(s[i0:i1])
                i0 = i1
                i1 = len(s)
            return r
        if type(texts)==list:
            R = [textseg(s) for s in texts]
            return R
        return textseg(texts)
def demo():
    texts = ["1》从左向右取待切分汉语句的m个字符作为匹配字段，m为大机器词典中最长词条个数。",
             "2》查找大机器词典并进行匹配。若匹配成功，则将这个匹配字段作为一个词切分出来。"]
    segment = textSegment('vocab_zh.txt')
    r = segment.textsSeg(texts)
    print(r)
if __name__=='__main__':
    demo()
                    
            
            
            