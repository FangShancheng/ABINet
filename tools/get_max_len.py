""" Created by MrBBS """
# 10/7/2022
# -*-encoding:utf-8-*-

print(max(open(r'E:\NLP_data\words_with_space.txt', 'r', encoding='utf-8').read().strip().split('\n'), key=len))
