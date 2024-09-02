def echo(message: str) -> str:
  print("------ C++ -> Python ------")
  print(message)
  print()
  return message + ", world!"

def main():
    txt = 'UMI SONODA'
    mes = echo(txt)
    print(mes)

if __name__ == '__main__':
  main()
