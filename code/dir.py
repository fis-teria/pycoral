import glob

def get_Image_Name(str):
    img_paths = glob.glob(str)
    for path in img_paths:
        print(path)

def main():
    str = input()
    get_Image_Name(str)

if __name__ == '__main__':
  main()
