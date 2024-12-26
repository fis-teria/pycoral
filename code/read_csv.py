import pandas as pd
import sys

argv = sys.argv

lst = pd.read_csv(argv[1])
#print(lst.sort_values("cosine_color", ascending=False))
print("------------------------------------------")
print(lst["cosine_color"].sort_values(ascending=False))
print("------------------------------------------")
print(lst["cosine_dedge"].sort_values(ascending=False))
print("------------------------------------------")
