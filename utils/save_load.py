import os

def delete_file(path):
    for i in range(len(path)):
        f = open(path[i],"w")
        f.close()
    return

def save_as_file(data,path):
    data = str(data)
    with open(path,"a") as f:
        f.write(data+'\n')
    return

def load_from_file(path):
    with open(path) as f:
        data = f.read().splitlines()
    data = list(map(float, data))
    return data