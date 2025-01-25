# https://qiita.com/5zm/items/49118188d76e61ca5113
# 渡されたファイルリストの順序で１つのファイルに結合する
def join_file(fileList, filePath):

    with open(filePath, 'wb') as saveFile:
        for f in fileList:
            data = open(f, "rb").read()
            saveFile.write(data)
            saveFile.flush()

import os, sys

def main():

    argv = sys.argv
    argc = len(argv)

    print('%s merges divided files' % argv[0])
    print('[usage] python %s <filename to be merged>' % argv[0])
    print('ex) python %s model_01.onnx' % argv[0])

    if argc < 2:
        quit()

    filePath = argv[1]

    no = 0
    fileList = []
    filename = filePath + '.%d' % no
    while os.path.exists(filename):
        fileList.append(filename)
        no += 1
        filename = filePath + '.%d' % no

    join_file(fileList, filePath)

if __name__ == '__main__':
    main()
