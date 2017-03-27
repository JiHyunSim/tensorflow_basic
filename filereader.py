#-*- coding:utf-8 -*-
#파일 읽어오는 함수
def read_csv_file(name,delimiter=',') :

    f = open(name,mode='r',encoding='UTF-8')

    #리턴배열
    res = []
    n=0
    length=0
    while True:
        #한줄 읽어온다.
        line = f.readline()
        if not line: break

        #뉴라인 제거
        line = line.replace('\n', '')
        #문자열이 없으면 다음으로
        if len(line) == 0:
            continue

        arr = line.split('|')
        length = max(length,len(arr))
        res.append(arr)
        n+=1

    f.close()
    print("read file :",name," is done. total rows :",n,", max columns :",length)

    return res,n
