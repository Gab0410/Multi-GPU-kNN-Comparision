import faiss
import numpy as np
from time import time   
import pandas as pd
from time import sleep


## Global variable
NORM = False
HNSW = False

#################################################################################

##                              DATASET FUNCTIONS
##
##          The functions above are related with the creation of the artificial 
##          datasets that are going to be used in the benchmarking

#################################################################################


## Join the arrays that have differente probabilistic distributions
def join_sample(data):
    
    
    sample = data[0]
    for i in range(1,len(data)):
        sample = np.concatenate((sample,data[i]))
    
    return sample

## Create the colors for each probabilistic distribution (importante for the future)
def make_colors(colors):
    
    sample = colors[0]
    max_c = max(colors[0])
    
    for i in range(1,len(colors)):
        colors[i] = colors[i] + max_c + 1   
        max_c = max(colors[i])
        sample = np.concatenate((sample,colors[i]))
    return sample

## Create a dataset with bicluster distributions
def biclust_dataset(N,dim):
    #Building make_bicluster dataset
    from sklearn.datasets import make_biclusters
    X0, rows,_ = make_biclusters(
    shape=(N, dim), n_clusters=2, noise=.4,minval=-12,maxval=10, shuffle=False, random_state=10)
    y0 = set_colors(rows,N) #Colors
    
    return X0,y0

## Create dataset with make_blobs distribution
def blobs_dataset(N,dim):
    #Building make_blobs dataset
    from sklearn.datasets import make_blobs
    X1, y1 = make_blobs(n_samples=N, centers=5, n_features=dim,
                   random_state=10,cluster_std=.6)
    return X1,y1

## Normalize the data, only if necessary
def normalize_dataset(data):
    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    norm_data = scaler.fit_transform(data)
    
    return norm_data

## Get the datasets with the propreties that is especified, and call the make col func
def get_artifical_db(N,dim):
    
    old_n = N
    N = N//2
    
    x0,y0 = biclust_dataset(N,dim)
    
    x1,y1 = blobs_dataset(N,dim)
    
    data = [x0,x1]
    colors = [y0,y1]
    
    sample = join_sample(data)
    col_list = make_colors(colors)
    

    if NORM:
        #É preciso normalizar o conjunto de dados, visto que a distância utilizada é a euclidiana
        normalized_sample = normalize_dataset(sample)
    else:
        normalize_dataset = sample
    
    np.random.shuffle(normalized_sample)
    return normalized_sample,col_list

## Create the dataset by calling the functions above and check their integrity
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

#################################################################################

##                              MEASUREMENT FUNCTIONS
##
##          The functions above are related with the measurement of the kNN methods
##          that are being tested in this benchmark

#################################################################################


## Calculate the recall@k of the method
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
    
## Calculate the mean of the kNN times of the methods
def analysis_runtime(values):
    
    values = np.array(values)

    mean = values.mean()
  
    
    
    return mean


#################################################################################

##                              kNN SEARCHING Class
##
##          The Class above are related with the kNN methods
##          that are being tested in this benchmark

#################################################################################

class MultiGPUIndex:

    #Construtor da classe
    def __init__(self, data, name):
        
        self.name = name
        self.data = data

    #Destrutor da classe
    def __del__(self):
        del self.data


class MultiBrute(MultiGPUIndex):

    def __init__(self, data, name):
        super().__init__(data, name)

        ## Multi-GPU config
        self.gpus = list(range(faiss.get_num_gpus()))
        self.res = [faiss.StandardGpuResources() for _ in self.gpus]
        self.co = faiss.GpuMultipleClonerOptions()

    def search(self,k):

        """
        This functions performs the search of the kNN and them return the indices of them
        """

        ## Shape of the data
        n, d = self.data.shape

        ## Make the index on CPU
        index_cpu = faiss.IndexFlatL2(d)


        ## Make the index on Multi-GPU
        index = faiss.index_cpu_to_gpu_multiple_py(self.res, index_cpu, self.co, self.gpus)


        t0 = time()

        ## Add the data to the index to have better performance
        index.add(self.data)

        ## Search
        _, I = index.search(self.data, k)
        
        tf = time() - t0
        return np.array(I),tf

class MultiIVFFlat(MultiGPUIndex):

    def __init__(self, data, name,quantizer,nprobe,nlist):
        super().__init__(data, name)

        ## Approximate method settings
        self.quantizer = quantizer
        self.nprobe = nprobe
        self.nlist = nlist

        ## Multi-GPU config
        self.gpus = list(range(faiss.get_num_gpus()))
        self.res = [faiss.StandardGpuResources() for _ in self.gpus]
        self.co = faiss.GpuMultipleClonerOptions()

    def __del__(self):
        del self.quantizer
        return super().__del__()

    def search(self,k):

        """
        This functions performs the search of the kNN and them return the indices of them
        """

        ## Data shape
        n, d = self.data.shape

        ## Creating the Index on CPU
        index_cpu = faiss.IndexIVFFlat(self.quantizer,d,self.nlist,faiss.METRIC_L2)

        ## Setting nprobe to the index
        index_cpu.nprobe = self.nprobe


        ## Making the Multi-GPU index
        index = faiss.index_cpu_to_gpu_multiple_py(self.res, index_cpu, self.co, self.gpus)


        t0 = time()
    
        ## Training the index
        index.train(self.data)

        ## Adding the data to the index
        index.add(self.data)

        ## Perform the search
        _, I = index.search(self.data, k)
        
        tf = time() - t0
        return np.array(I),tf

class MultiIVFPQ(MultiGPUIndex):

    def __init__(self, data, name,quantizer,nprobe,nlist,M):
        super().__init__(data, name)

        ## Approximate method settings
        self.quantizer = quantizer
        self.nprobe = nprobe
        self.nlist = nlist
        self.M = M

        ## Multi-GPU config
        self.gpus = list(range(faiss.get_num_gpus()))
        self.res = [faiss.StandardGpuResources() for _ in self.gpus]
        self.co = faiss.GpuMultipleClonerOptions()

    def __del__(self):
        del self.quantizer
        return super().__del__()

    def search(self,k):

        """
        This functions performs the search of the kNN and them return the indices of them
        """

        ## Data shape
        n, d = self.data.shape

        ## Making the index on CPU
        index_cpu = faiss.IndexIVFPQ(self.quantizer,d,self.nlist,self.M,faiss.METRIC_L2)

        ## Setting nprobe to the index
        index_cpu.nprobe = self.nprobe


        ## Making the Multi-GPU index
        index = faiss.index_cpu_to_gpu_multiple_py(self.res, index_cpu, self.co, self.gpus)


        t0 = time()
    
        ## Training the index
        index.train(self.data)

        ## Adding the data to the index
        index.add(self.data)

        ## Perform the search
        _, I = index.search(self.data, k)
        
        tf = time() - t0
        return np.array(I),tf
    
class MultiIVFSQ(MultiGPUIndex):

    def __init__(self, data, name,quantizer,nprobe,nlist):
        super().__init__(data, name)

        ## Approximate method settings
        self.quantizer = quantizer
        self.nprobe = nprobe
        self.nlist = nlist
        self.qtype = faiss.ScalarQuantizer.QT_8bit

        ## Multi-GPU config
        self.gpus = list(range(faiss.get_num_gpus()))
        self.res = [faiss.StandardGpuResources() for _ in self.gpus]
        self.co = faiss.GpuMultipleClonerOptions()
    
    def __del__(self):
        del self.quantizer
        return super().__del__()




    def search(self,k):

        """
        This functions performs the search of the kNN and them return the indices of them
        """

        ## Data shape
        n, d = self.data.shape


        ## Making the index on CPU
        index_cpu = faiss.IndexIVFScalarQuantizer(self.quantizer,d,self.nlist,self.qtype,faiss.METRIC_L2)

        ## Setting nprobe to the index
        index_cpu.nprobe = self.nprobe


        ## Making the Multi-GPU index
        index = faiss.index_cpu_to_gpu_multiple_py(self.res, index_cpu, self.co, self.gpus)


        t0 = time()
    
        ## Training the index
        index.train(self.data)

        ## Adding the data to the index
        index.add(self.data)

        ## Perform the search
        _, I = index.search(self.data, k)
        
        tf = time() - t0
        return np.array(I),tf

#################################################################################

##                              AUXILIAR FUNCTIONS
##
##          The function below are related with auxiliar methods

#################################################################################

def write_df(df,index,name,method,dim,n_sample,time_knn,rec_value,k,nlist,nprobe):
    df.loc[index, 'Name'] = name
    df.loc[index, 'Method'] = method
    df.loc[index, 'Dim'] = dim
    df.loc[index, 'N_sample'] = n_sample
    df.loc[index, 'Time kNN'] = time_knn
    df.loc[index, f'Recall@{k}'] = rec_value
    df.loc[index,'nList'] = nlist
    df.loc[index,'nprobe'] = nprobe

    return


def instanciate_dataset(info):

    """
    info structure: info = {'n_range':(a,b),'d_range':(a,b),'n_d':a,'decrement':b}

    N range is the range of the n_sample values (a > b)
    d_range is the range of the dimensions values (a > b)
    n_d is the especific n_sample value, that the number of dimensions will be varied, this can only be one n_sample value
    decrement is how much each iteration will decrement the value of n_sample
    """

    dbs = {}
    start = 'SK-'

    n_start,n_end = info['n_range']
    d_start,d_end = info['d_range']

    especific_n = info['n_d']
    decrement = -info['decrement']

    for i in range(n_start,n_end-1,decrement):
        number = str( ( i / 1e6) )
        base_name = start + number+ 'M-'

        name = base_name + str(d_start) + 'd'

        
        if i == especific_n:
            for j in range(d_start,d_end-1,-2):
                name = base_name + str(j) + 'd'

                dbs[name] = create_dataset(i,j)[0]
        else:
            dbs[name] = create_dataset(i,d_start)[0] 
    
    return dbs

def create_object(name,info):

    index = None

    ## Create brute force index
    if name == 'brute':
        index = MultiBrute(info['data'],name)
        return index

    ## Create quantizer for the approximate METHODS
    quantizer = None
    D = info['data'].shape[1]
    hnsw_m = 32  

    if not HNSW:
        quantizer = faiss.IndexFlatL2(D)
    else:
        quantizer = faiss.IndexHNSWFlat(D, hnsw_m)

    ## Create IVFFlat index
    if name == 'ivfflat':
        index = MultiIVFFlat(info['data'],name,quantizer,info['nprobe'],info['nlist'])

    ## Create IVFPQ index
    elif name == 'ivfpq':

        M_list = [1, 2, 3, 4, 8, 12, 16, 20, 24, 28, 32, 48, 56, 64, 96] 
        M = D
        count = len(M_list)-1
        while M % M_list[count] != 0:
            count -= 1
            if count == 0:
                break
        M = M_list[count]

        index = MultiIVFPQ(info['data'],name,quantizer,info['nprobe'],info['nlist'],M)
    
    ## Create IVFSQ index
    else:
        index = MultiIVFSQ(info['data'],name,quantizer,info['nprobe'],info['nlist'])
    
    return index

#################################################################################

##                              PRIMARY EXECUTION
##
##          Above the primary execution is happening

#################################################################################


## Methods
metodos = {'deterministico': ['brute'],
           'probabilistico':['ivfflat',
                            'ivfpq',
                            'ivfsq']
}

## Dataset informations
info_dbs = {'n_range':(int(1e7),int(1e6)),
            'd_range':(12,2),
            'n_d':int(5e6),
            'decrement':int(1e6)}

# Create the datasets and load the to RAM
dbs = instanciate_dataset(info_dbs)


def main():

    ## Initializating the dataframe
    df_gpu = df_gpu = pd.DataFrame()

    ## Control variable
    index = 0

    ## Order strategy
    order_s = 'recall'

    ## Setting K value
    k = 20

    ## Setting recall@K value
    rec_k = 10

    ## Iterate the datasets
    for db in dbs:

        ## Create the variable to save the exact result
        brute_indices = None

        for c in metodos: #deterministico e probabilistico
            for method in metodos[c]: #brute, ivfflat,ivfpq,ivfsq

                ## Warm-UP GPU
                if index == 0:
                    print("Warming up the GPU...")
                    _,_ = MultiBrute(db,method).search(k)
                    print("GPU Ready to go...")
                    sleep(5)
                
                ## Writing control variable
                index += 1

                #List if the are repetitions
                results = []

                """
                Here, all the possible configurations that the approximate methods can be executed will be tested, and their results will be inserted
                on the list above. It will be possible to order the results by time or recall
                """

                ## Setting nlists and nprobes values for the approximate method
                nlists = [10000]
                nprobes = [10]

                ## Setting random values for brute method
                if method == 'brute':
                    nlists = [0]
                    nprobes = [0]

                ## Info declaration
                info = {}

                ## Primary iteration
                for nlist in nlist:
                    for nprobe in nprobes:

                        ## Set the information about the methdo, if the brute method is performed nlist and nprobe will not be used
                        info['nlist'] = nlist
                        info['nprobe'] = nprobe
                        info['data'] = db

                        ## Create the object by the name 
                        index = create_object(method,info)

                        ## Perform the search, saving the indices and the kNN time
                        indices,time_knn = index.search(k)
                        
                        rec_value = '-'

                        ## Save the exact result
                        if method == 'brute':
                            brute_indices = indices.copy()
                            result = {'time_knn':time_knn,'recall':'-','nlist':'-','nprobe':'-'}

                        ## Calculate the results and ADD them to the list
                        if method != 'brute':
                            rec_value = recall(brute_indices,indices,rec_k)
                            result = {'time_knn':time_knn,'recall':rec_k,'nlist':nlist,'nprobe':nprobe}
                            results.append(result)  
                
                ## Set the strategy name
                strategy = 'recall'
                if order_s == 'time':
                    strategy = 'time_knn'
                
                ## Order the results by a strategy that is decide by the user
                if method != 'brute':
                    result = sorted(result,key=lambda d:d[strategy],reverse=True)

            #Save the results in a dataframe
            time_knn = result[0]['time_knn']
            rec_value = result[0]['recall']
            nlist = result[0]['nlist']
            nprobe = result[0]['nprobe']
            n_sample,dim = db.shape
            write_df(df_gpu,index,db,method,dim,n_sample,time_knn,rec_value,rec_k,nlist,nprobe)

            #Show the informations to check how the test is going and in what stage the test is on
            print(f"Iteration -> {index} DB -> {db} Dim -> {dim} N -> {n_sample} Finished in {time_knn:.5} secs, method -> {method}, recall -> {rec_value}")            


    ## Write DataFrame 
    df_gpu.to_csv('raw_data_gpu.csv', index=False)


main()
# https://github.com/matsui528/faiss_tips -> Faiss Tips
# Error in classes https://github.com/facebookresearch/faiss/issues/540
# Memory issue https://github.com/facebookresearch/faiss/wiki/FAQ#how-can-i-set-nprobe-on-an-opaque-index




