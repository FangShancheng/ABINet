chars = open(r'D:\SourceCode\bill_extract\crawl_data\char.txt', 'r', encoding='utf-8').read().strip().split('\t')

with open('charset_vn.txt', 'w', encoding='utf-8') as f:
    for i, char in enumerate(chars):
        print(char)
        f.write(f'{i}\t{char}\n')
