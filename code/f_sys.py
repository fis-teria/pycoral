def f_open(file):
    import sys
    data = []
    try:
        f = open(file, 'r', encoding='utf-8')
    except Exception:
        print("open error. not found file:", str(file))
        sys.exit(1)
    for line in f:
        line = line.strip() #前後空白削除
        line = line.replace('\n','') #末尾の\nの削除
        line = line.split(",") #分割
        data.append(line)
    f.close()
    return data

def f_index(l, str):
    if [str] in l:
        return l.index([str])
    else:
        return 0
def main():
    data = []
    data = f_open("../pycoral/test_data/pascal_voc_segmentation_labels.txt")
    print(data)
    print(data.index(['person']))
    print(f_index(data, 'person'))

if __name__ == '__main__':
  main()
