import numpy as np
from time import time
import faiss                     # make faiss available


#Concatena as distribuições probabilísticas
def make_sample(data):
    
    
    sample = data[0]
    for i in range(1,len(data)):
        sample = np.concatenate((sample,data[i]))
    
    
    return sample

def make_colors(colors):
    
    sample = colors[0]
    max_c = max(colors[0])
    
    for i in range(1,len(colors)):
        colors[i] = colors[i] + max_c + 1   
        max_c = max(colors[i])
        sample = np.concatenate((sample,colors[i]))
    return sample

def biclust_dataset(N,dim):
    #Building make_bicluster dataset
    from sklearn.datasets import make_biclusters
    X0, rows,_ = make_biclusters(
    shape=(N, dim), n_clusters=2, noise=.4,minval=-12,maxval=10, shuffle=False, random_state=10)
    y0 = set_colors(rows,N) #Colors
    
    return X0,y0

def blobs_dataset(N,dim):
    #Building make_blobs dataset
    from sklearn.datasets import make_blobs
    X1, y1 = make_blobs(n_samples=N, centers=5, n_features=dim,
                   random_state=10,cluster_std=.6)
    return X1,y1

def normalize_dataset(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(data)
    
    return norm_data

def get_artifical_db(N,dim):
    
    old_n = N
    N = N//2
    
    x0,y0 = biclust_dataset(N,dim)
    
    x1,y1 = blobs_dataset(N,dim)
    
    data = [x0,x1]
    colors = [y0,y1]
    
    sample = make_sample(data)
    col_list = make_colors(colors)
    
    #É preciso normalizar o conjunto de dados, visto que a distância utilizada é a euclidiana
    normalized_sample = normalize_dataset(sample)
    np.random.shuffle(normalized_sample)
    return normalized_sample,col_list

def create_dataset(N,dim):
    
    sample,col_list = get_artifical_db(N,dim)
    colors = col_list
    N = sample.shape[0]
    i0 = 0
    for i in range(N//2,len(colors),N):
        
        c_unique = colors[i0:i]
        c_out = colors[i:]
        
        unique = np.sort(pd.unique(c_unique))
        unique_out = np.sort(pd.unique(c_out))
        
        i0 = i
        
        for i in unique:
            if i in unique_out:
                print(f"O valor {i} esta na lista {unique_out}")
                exit()
      
    return sample.astype(np.float32),col_list
"""
dbs = {'SK-1M-2d': create_dataset(int(1e6),2)[0],
        'SK-1M-20d':create_dataset(int(1e6),20)[0],
        'SK-1M-32d':create_dataset(int(1e6),32)[0],
        'SK-1M-12d':create_dataset(int(1e6),12)[0],
        'SK-5M-12d':create_dataset(int(5e6),12)[0],
        'SK-10M-12d':create_dataset(int(10e6),12)[0],
        'SK-13M-12d':create_dataset(int(13e6),12)[0],
      }
"""
def recall(arr1,arr2,k):
    
    #Verificação da integridade
    if arr1.shape != arr2.shape:
        print("Impossível de fazer a avaliação, as arrays tem tamanho diferentes")
    elif arr1.shape[1] < k:
        print(f"Impossível de fazer o recall{k}, já que as array não tem {k} vizinhos")
    
    #Somatório dos k primeiros vizinhos positivos dividido por n*k
    acertos = 0
    
    n = arr1.shape[0]


    recall_value = (arr1[:,:k] == arr2[:,:k]).sum() / (float(n*k))
    
    return recall_value


ngpus = faiss.get_num_gpus()

print("number of GPUs:", ngpus)

"""
t0 = time()
cpu_index = faiss.IndexFlatL2(d)

gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
    cpu_index
)

gpu_index.add(xb)              # add vectors to the index
print(gpu_index.ntotal)

k = 20                          # we want to see 4 nearest neighbors
_, I_e = gpu_index.search(xb, k) # actual search
print(f"Tempo exato= {time()-t0}")
"""

def hehe(n,dim):
    d = dim                          # dimension
    nb = int(n)                      # database size
    np.random.seed(1234)             # make reproducible
    xb = np.random.random((nb, d)).astype('float32')
    
    return xb

nlist = range(int(1e5),int(5e3)-1,-5000)
configs_err = {}
nprobes = range(4,300,5)



N = range(int(1e8),int(1e6)-1,-int(5e6))
d = 12


k = 20

for p in N:

    xb = hehe(p,d)
    key_atual = p
    configs_err[key_atual] = {}
    for i in range(len(nlist)):

        qntd = faiss.IndexFlatL2(d)
        for j in nprobes:
            
            t0 = time()
            
            cpu_index = faiss.IndexIVFFlat(qntd,d,nlist[i],faiss.METRIC_L2) 
            cpu_index.nprobe = j
            print(f"Nlist = {nlist[i]} nprobe = {j}")
            print("Fazendo o indice")
            gpu_index = faiss.index_cpu_to_all_gpus(  # build the index
            cpu_index
            )


            try:
                print("Treinando o negocio")
                gpu_index.train(xb)
                gpu_index.add(xb)              # add vectors to the index
                print(f"Tempo de treino = {time()-t0}")
                print("Fazendo a busca")
                _, I = gpu_index.search(xb, k) # actual search
                print(f"Tempo aproximado = {time()-t0}")
                del I
            except Exception as e:
                if nlist[i] in configs_err[key_atual]:
                    configs_err[key_atual][nlist[i]].append(j)
                else:
                    configs_err[key_atual][nlist[i]] = [j]
                print(f"ERRO tamanho = {p} nlist = {nlist[i]} & nprobe = {j}")
                del cpu_index,gpu_index
                break

            del cpu_index,gpu_index

        del qntd

    del xb
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

write_dict(configs_err)
print(dict)
"""
    rec = recall(I_e,I,5)
    print(f"Recall@5 = {rec}")


    rec = recall(I_e,I,10)
    print(f"Recall@10 = {rec}")

    rec = recall(I_e,I,20)
    print(f"Recall@20 = {rec}")

"""