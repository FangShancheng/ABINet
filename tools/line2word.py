""" Created by MrBBS """
# 10/4/2022
# -*-encoding:utf-8-*-

from tqdm import tqdm
import random

lines = list(set(open(r'E:\NLP_data\corpus_data.txt', 'r', encoding='utf-8').read().strip().split('\n') +
                 open(r'D:\SourceCode\bill_extract\crawl_data\english_book.txt', 'r',
                      encoding='utf-8').read().strip().split('\n')))
chars = ''.join(
    open(r'D:\SourceCode\bill_extract\crawl_data\char.txt', 'r', encoding='utf-8').read().strip().split('\t'))
words = set()

max_num_words = 3
min_num_words = 1
max_len = 0

for line in tqdm(lines):
    w = line.strip().split(' ')
    num_words = random.randint(min_num_words, max_num_words)
    for i in range(0, len(w), num_words):
        word = ' '.join(w[i:i + num_words]).strip()
        len_word = len(word)
        if len_word > 1 and set(word).issubset(chars):
            words.add(word)
        max_len = max(max_len, len_word)
    # for word in line.strip().split(' '):
    #     if 51 > len(word) > 1 and set(word).issubset(chars):
    #         words.add(word)
open(r'E:\NLP_data\words_with_space.txt', 'w', encoding='utf-8').write('\n'.join(words))
open('../data/max_len.txt', 'w').write(str(max_len))
