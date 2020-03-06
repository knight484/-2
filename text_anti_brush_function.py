import numpy as np
import jieba
from simhash import Simhash
import re


# 定义去掉图片html标签函数
def filter_html(html):
    """
    :param html: html
    :return: 返回去掉html的纯净文本
    """
    dr = re.compile(r'<[^>]+>', re.S)
    dd = dr.sub('', html).strip()
    return dd



# 句子文本转换成向量
def get_word_vector(s1, s2):
    """
    :param s1: 句子1
    :param s2: 句子2
    :return: 返回句子的余弦相似度
    """
    # 分词
    cut1 = jieba.cut(s1)
    cut2 = jieba.cut(s2)
    list_word1 = (','.join(cut1)).split(',')
    list_word2 = (','.join(cut2)).split(',')

    # 列出所有的词,取并集
    key_word = list(set(list_word1 + list_word2))
    # 给定形状和类型的用0填充的矩阵存储向量
    word_vector1 = np.zeros(len(key_word))
    word_vector2 = np.zeros(len(key_word))

    # 计算词频
    # 依次确定向量的每个位置的值
    for i in range(len(key_word)):
        # 遍历key_word中每个词在句子中的出现次数
        for j in range(len(list_word1)):
            if key_word[i] == list_word1[j]:
                word_vector1[i] += 1
        for k in range(len(list_word2)):
            if key_word[i] == list_word2[k]:
                word_vector2[i] += 1

    # 输出向量
    # print(word_vector1)
    # print(word_vector2)
    return word_vector1, word_vector2


# 计算两个向量之间的余弦相似度
def cos_dist(vec1, vec2):
    """
    :param vec1: 向量1
    :param vec2: 向量2
    :return: 返回两个向量的余弦相似度
    """
    dist1 = float(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2)))
    return dist1




# 定义求两个短句子的相似度函数
def bow_similariry(text1,text2):
    """
    :param text1: 文本1
    :param text2: 文本2
    :return: 余弦相似度
    """
    text1 = filter_html(text1)
    text2 = filter_html(text2)

    vec1, vec2 = get_word_vector(text1, text2)
    dist=cos_dist(vec1, vec2)
    return dist


# 定义求两篇文章相似度函数
def simhash_similarity(text1, text2):
    """
    :param tex1: 文本1
    :param text2: 文本2
    :return: 返回两篇文章的相似度
    """
    text1 = filter_html(text1)
    text2 = filter_html(text2)
    aa_simhash = Simhash(text1)
    bb_simhash = Simhash(text2)
    max_hashbit = max(len(bin(aa_simhash.value)), (len(bin(bb_simhash.value))))
    # 汉明距离
    distince = aa_simhash.distance(bb_simhash)
    # 相似度计算
    similar = 1 - distince / max_hashbit
    return similar





# 定义判断句子是长文本或者短文本，并计算相似度
def similarity_classification(text1,text2):
    """
    :param text1: 文本1
    :param text2: 文本2
    :return: 返回相似度，短文本用tf-idf,长文本用simhash
    """
    len_1=len(text1)
    len_2=len(text2)
    # print(len_1, len_2)
    max_len=max(len_1,len_2)


    if max_len<50:

        return bow_similariry(text1,text2)
    else:
        return simhash_similarity(text1,text2)



def LCS(x,y):

    c=np.zeros((len(x)+1,len(y)+1))
    b=np.zeros((len(x)+1,len(y)+1))
    for i in range(1,len(x)+1):
        for j in range(1,len(y)+1):
            if x[i-1]==y[j-1]:
                c[i,j]=c[i-1,j-1]+1
                b[i,j]=2
            else:
                if c[i-1,j]>=c[i,j-1]:
                    c[i,j]=c[i-1,j]
                    b[i,j]=1
                else:
                    c[i,j]=c[i,j-1]
                    b[i,j]=3
    return c,b



# 最长公共子串
def getLCS(x,y):
    c,b=LCS(x,y)
    i=len(x)
    j=len(y)
    lcs=''
    while i>0 and j>0:
        if b[i][j]==2:
            lcs=x[i-1]+lcs
            i-=1
            j-=1
        if b[i][j]==1:
            i-=1
        if b[i][j]==3:
            j-=1
        if b[i][j]==0:
            break
    lcs_len=len(lcs)
    return lcs,lcs_len



# 最长公共子串相似度
def lcs_similarity(a,b):
    lcs, lcs_len = getLCS(a, b)
    similarity = round(lcs_len / max(len(a), len(b)), 4)
    return similarity
