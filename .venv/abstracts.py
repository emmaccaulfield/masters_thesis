import bz2
file_name = "/mnt/c/Users/emmaclai/Documents/Master/thesis/short-abstracts_lang=es.ttl.bz2"
f = bz2.open(file_name,mode="rt")

for line in f:
  print(line)

# print(f[0])
# print(f[1])