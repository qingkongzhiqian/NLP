import Levenshtein
import jieba
from nltk.metrics.distance import jaccard_distance
import math
import jieba.analyse
import jieba.posseg as pseg
import codecs
from gensim import corpora,models,similarities
from datasketch import MinHash

class SimHash(object):
    def __init__(self):
        pass
    def getBinStr(self, source):
        if source == "":
            return 0
        else:
            x = ord(source[0]) << 7
            m = 1000003
            mask = 2 ** 128 - 1
            for c in source:
                x = ((x * m) ^ ord(c)) & mask
            x ^= len(source)
            if x == -1:
                x = -2
            x = bin(x).replace('0b', '').zfill(64)[-64:]
            return str(x)

    def getWeight(self, source):
        # fake weight with keyword
        return ord(source)
    def unwrap_weight(self, arr):
        ret = ""
        for item in arr:
            tmp = 0
            if int(item) > 0:
                tmp = 1
            ret += str(tmp)
        return ret

    def simHash(self, rawstr):
        seg = jieba.cut(rawstr)
        keywords = jieba.analyse.extract_tags("|".join(seg), topK=100, withWeight=True)
        ret = []
        for keyword, weight in keywords:
            binstr = self.getBinStr(keyword)
            keylist = []
            for c in binstr:
                weight = math.ceil(weight)
                if c == "1":
                    keylist.append(int(weight))
                else:
                    keylist.append(-int(weight))
            ret.append(keylist)
        # 对列表进行"降维"
        rows = len(ret)
        cols = len(ret[0])
        result = []
        for i in range(cols):
            tmp = 0
            for j in range(rows):
                tmp += int(ret[j][i])
            if tmp > 0:
                tmp = "1"
            elif tmp <= 0:
                tmp = "0"
            result.append(tmp)
        return "".join(result)

    def getDistince(self, hashstr1, hashstr2):
        length = 0
        for index, char in enumerate(hashstr1):
            if char == hashstr2[index]:
                continue
            else:
                length += 1
        return length

s1 = "kitten"
s2 = "sitting"

def get_lcs1(str1,str2):
    len1,len2 = len(str1),len(str2)
    mat = [[0 for i in range(len2+1)]for j in range(len1+1)]
    for i in range(len1+1):
        for j in range(len2+1):
            if i == 0 or j == 0:
                mat[i][j] = 0
            elif str1[i-1] == str2[j-1]:
                mat[i][j] = mat[i-1][j-1]+1
            else:
                mat[i][j] = max(mat[i - 1][j], mat[i][j - 1])
    return mat[-1][-1]

raw_documents = [
   "这只皮靴号码大了。那只号码合适。",
    "这只皮靴号码不小，那只更合适。"
]

corpora_documents = []
for item_text in raw_documents:
    item_str = jieba.lcut(item_text)
    corpora_documents.append(item_str)

# 生成字典和向量语料
dictionary = corpora.Dictionary(corpora_documents)
corpus = [dictionary.doc2bow(text) for text in corpora_documents]

similarity = similarities.Similarity('-Similarity-index', corpus, num_features=400)

test_data_1 = '这只皮靴号码大了。那只号码合适'
test_cut_raw_1 = jieba.lcut(test_data_1)
test_corpus_1 = dictionary.doc2bow(test_cut_raw_1)
similarity.num_best = 5
# print(similarity[test_corpus_1]) # 返回最相似的样本材料,(index_of_document, similarity) tuples

def dice_coefficient(a, b):
    a_bigrams = set(a)
    b_bigrams = set(b)
    overlap = len(a_bigrams & b_bigrams)
    print ("overlap",overlap)
    print(len(a_bigrams))
    print(len(b_bigrams))
    return overlap * 2.0/(len(a_bigrams) + len(b_bigrams))

simhash = SimHash()
s1_sim = u'I am very happy'
s2_sim = u'I am very happu'

hash1 = simhash.simHash(s1_sim)
hash2 = simhash.simHash(s2_sim)
distince = simhash.getDistince(hash1, hash2)
value = 5

data1 = ['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'datasets']

data2 = ['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for',
        'estimating', 'the', 'similarity', 'between', 'documents']

m1,m2 = MinHash(),MinHash()

for d in data1:
        m1.update(d.encode("utf8"))
for d in data2:
        m2.update(d.encode("utf8"))
#minhash之后的文本相似度
print ("Estimated jaccard for data1 and data2 is",m1.jaccard(m2))

#原始data的文本相似度
from nltk.metrics.distance import jaccard_distance
print ("Origin jaccard similarity",1 - jaccard_distance(set(data1),set(data2)))

#Minhash LSH
from datasketch import MinHash,MinHashLSH
set1 = set(['minhash', 'is', 'a', 'probabilistic', 'data', 'structure', 'for','estimating', 'the', 'similarity', 'between', 'datasets'])
set2 = set(['minhash', 'is', 'a', 'probability', 'data', 'structure', 'for','estimating', 'the', 'similarity', 'between', 'documents'])
set3 = set(['minhash', 'is', 'probability', 'data', 'structure', 'for','estimating', 'the', 'similarity', 'between', 'documents'])

m1 = MinHash(num_perm=128)
m2 = MinHash(num_perm=128)
m3 = MinHash(num_perm=128)
for d in set1:
    m1.update(d.encode('utf8'))
for d in set2:
    m2.update(d.encode('utf8'))
for d in set3:
    m3.update(d.encode('utf8'))

#Create LSH index
lsh = MinHashLSH(threshold=0.5,num_perm=128)
lsh.insert("m2",m2)
lsh.insert("m3",m3)
result = lsh.query(m1)
print ("lsh",lsh)
print ("Approximate neighbours with Jaccard similarity > 0.5", result)

print ("编辑距离",Levenshtein.distance(s1,s2))
print ("汉明距离",Levenshtein.hamming("karolin","kathrin"))
print ("最长公共子序列",get_lcs1('Hello World','Bonjour le monde'))
print ("jaro",Levenshtein.jaro('aboard','aborad'))
print ("jaro_winkler",Levenshtein.jaro_winkler('aboard','aborad'))
print ("Jaccard",jaccard_distance(set(['这只','皮靴','号码','大了','那只','号码','合适']),set(['这只','皮靴','号码','不','小','那只','更','合适'])))
print ("dice",dice_coefficient(['这只','皮靴','号码','大了','那只','号码','合适'],['这只','皮靴','号码','不','小','那只','更','合适']))
print("海明距离：", distince, "判定距离：", value, "是否相似：", distince<=value)
