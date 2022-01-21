'''
Charset에 문자열을 추가해주는 코드
실행은 python create_charset.py로 한다.
'''

def writeCharset(startIdx, charset):
    outList = []
    if startIdx is not 0:        
        outList.append('\n') # 이어서 계속 해나갈 때는 맨처음에 줄바꿈을 넣어준다.
        
    for i, Char in enumerate(charset):       
        indexChar = "%d\t%s\n"%(i+startIdx, Char) # 인덱스 번호 / 탭 / 글자 / 줄바꿈
        outList.append(indexChar)
    outList[-1] = outList[-1].replace('\n', '') # 마지막 줄의 줄바꿈 문자 제거
    return outList


def writeFile(outPathfile, outList):
    with open(outPathfile, 'w', encoding='utf-8') as wf:
        wf.writelines(outList)
        print(outPathfile + " saved")

        
if __name__=="__main__":
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    digit = "1234567890"
    hangul = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣㄳㄵㄶㄺㄻㄼㄽㄾㄿㅀㅄ"
    gana = "ーぁあぃいぅうぇえぉおかがきぎくぐけげこごさざしじすずせぜそぞただちぢっつづてでとどなにぬねのはばぱひびぴふぶぷへべぺほぼぽまみむめもゃやゅゆょよらりるれろゎ わゐゑをんァアィイゥウェエォオカガキギクグケゲコゴサザシジスズセゼソゾタダチヂッツヅテデトドナニヌネノハバパヒビピフブプヘベペホボポマミムメモャヤュユョヨラリルレロヮワヰヱヲンヴヵヶ゛-゜"
    chinesechar= "生新辛綠紅黒黃茶牛肉安城湯麵台湾鳳梨酥名物"
    
    charSet = writeCharset(0, alphabet)
    charSet = charSet + writeCharset(len(alphabet), digit) # 인덱스 번호가 이어지도록 누적한다.
    charSet = charSet + writeCharset(len(alphabet)+len(digit), hangul) # 인덱스 번호가 이어지도록 누적한다
    
    writeFile("/home/ubuntu/Playground/ABINet/data/charset_36hangul.txt", charSet)