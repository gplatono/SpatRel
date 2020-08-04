def checkNot(str):
    str = str.split(' ')
    if str[0] == 'not':
        str.pop(0)
    str = ' '.join(str)
    return str

def checkLabel():
    dataList = ['RW1.data', 'RW2.data', 'RW3.data', 'RW4.data', 'RW5.data',
                 'RW6.data', 'RW7.data', 'RW8.data', 'RW9.data', 'RW10.data',
                 'RW11.data', 'RW101.data', 'RW201.data', 'RW202.data', 'RW203.data',
                 'RW204.data', 'RW205.data', 'RW206.data', 'RW207.data', 'RW208.data',
                 'RW209.data', 'RW210.data']
    labelList = []
    for fname in dataList:
        with open(fname, 'r') as f:
            lines = [line for line in f]
            #print(lines)
            for i, ele in enumerate(lines):
                if ele.strip() == '':
                    continue
                datalabel = ele.split(':')[1]
                datalabel = datalabel.strip()
                datalabel = checkNot(datalabel)
                if datalabel == 'east of':
                    print(f'{datalabel} found in line {i} in file ' + fname)
                if datalabel not in labelList:
                    labelList.append(datalabel)
    for label in labelList:
        print(label)

if __name__ == '__main__':
    checkLabel()

