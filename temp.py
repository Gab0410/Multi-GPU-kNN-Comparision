def write_dict(dic):
    file_name = 'test.txt'
    file = open(file_name,'w+')

    file.write("Resultados:\n\n")
    for j in dic:
        file.write(f"Numero de Amostras {j}: \n\n")
        for i in dic[j]:
            #print(i)
            file.write(f"{i} : A partir de {dic[j][i]} não funciona\n")
        file.write("\n")

    file.close()

def read_file():
    file_name = 'test.txt'
    file = open(file_name,'r+')

    dic = {}

    FLAG = 0
    key = 0
    for l in file:

        if l.startswith("Numero"):
            key = int(l.split(" ")[-2][:-1])
            dic[key] = {}
            FLAG = 1

        #Está dentro do bagulo
        elif FLAG:
            key_list = 0
            try:
                l_split = l.split(" ")
                key_list = int(l_split[0])
                lista = list(l_split[5])
                nprobe = ''

                for j in lista:
                    try:
                        aux = int(j)
                        nprobe += j
                        
                    except:
                        continue
                
                dic[key][key_list] = [int(nprobe)]
            except:
                continue
        
    file.close()

    return dic
    

dic = { 100: {'8000': [5],
        '9000': [2],
        '10000': [9] },
    
        200: {'5000': [12],
        '6000': [15],
        '70000': [18] }
    
}

#write_dict(dic)
d = read_file()

print(d)
print(dic)