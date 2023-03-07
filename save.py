import argparse
import os
import pickle
import time
from multiprocessing.pool import ThreadPool

import faiss
import numpy as np


def set_index_parameter(index, name, val):
    """
    Index parameter setting that works on the index lookalikes defined above
    """
    if index.__class__ == ShardedGPUIndex:
        if name == "nprobe":
            set_index_parameter(index.cpu_index, name, val)
        elif name.startswith("quantizer_"):
            set_index_parameter(
                index.quantizer, name[name.find("_") + 1:], val)
        else:
            raise RuntimeError()
        return

    # then it's a Faiss index
    index = faiss.downcast_index(index)

    if isinstance(index, faiss.IndexPreTransform):
        set_index_parameter(index.index, name, val)
    elif (isinstance(index, faiss.IndexShards) or
          isinstance(index, faiss.IndexReplicas)):
        for i in range(index.count()):
            sub_index = index.at(i)
            set_index_parameter(sub_index, name, val)
    elif name.startswith("quantizer_"):
        index_ivf = extract_index_ivf(index)
        set_index_parameter(
            index_ivf.quantizer, name[name.find("_") + 1:], val)
    elif name == "efSearch":
        index.hnsw.efSearch
        index.hnsw.efSearch = int(val)
    elif name == "nprobe":
        index_ivf = extract_index_ivf(index)
        index_ivf.nprobe
        index_ivf.nprobe = int(val)
        print("deu certo",index_ivf)
    else:
        raise RuntimeError(f"could not set param {name} on {index}")




def extract_index_ivf(index):
    """ extract the IVF sub-index from the index, supporting GpuIndexes
    as well """
    try:
        return faiss.extract_index_ivf(index)
    except RuntimeError:
        if index.__class__ == faiss.IndexPreTransform:
            index = faiss.downcast_index(index.index)
        if isinstance(index, faiss.GpuIndexIVF):
            return index
        raise RuntimeError(f"could not extract IVF index from {index}")


def search_preassigned(xq, k, index, quantizer, batch_size=0):
    """
    Explicitly call the coarse quantizer and the search_preassigned
    on the index.
    """

    #Shape da amostra de query
    n, d = xq.shape

    #nprobe
    nprobe = index.nprobe

    #Batch size não pode ser igual a 0, logo será o tamanho de + 1, sinalizado que haverá apenas 1 batch de tamanho igual ao conjunto de dados
    if batch_size == 0:
        batch_size = n + 1
    
    #Inicializa as arrays que irão conter os vizinhos e distâncias
    D = np.empty((n, k), dtype='float32')
    I = np.empty((n, k), dtype='int64')

    #Laço iterativo da batch
    for i0 in range(0, n, batch_size):
        #Faz a busca da batch
        Dq, Iq = quantizer.search(xq[i0:i0 + batch_size], nprobe)

        #Adiciona em D e em I, utilizando o método do Faiss
        D[i0:i0 + batch_size], I[i0:i0 + batch_size] = \
            index.search_preassigned(xq[i0:i0 + batch_size], k, Iq, Dq)
    
    #Retorna as distâncias da batch encontrada
    return D, I


class ShardedGPUIndex:
    """
    Multiple GPU indexes, each on its GPU, with a common coarse quantizer.
    The Python version of IndexShardsIVF
    """

    #Construtor da classe
    def __init__(self, quantizer, index, bs=-1, seq_tiling=False):
        self.quantizer = quantizer
        self.cpu_index = index

        #Caso seja índice do tipo Faiss PreTransform (descobrir oque é)
        if isinstance(index, faiss.IndexPreTransform):
            index = faiss.downcast_index(index.index)
        ngpu = index.count()
        self.pool = ThreadPool(ngpu)
        self.bs = bs
        if bs > 0:
            self.q_pool = ThreadPool(1)

    #Destrutor da classe
    def __del__(self):
        self.pool.close()
        if self.bs > 0:
            self.q_pool.close()

    #FUnção principal da classe que faz a busca
    def search(self, xq, k):
        nq = len(xq)
        # perform coarse quantization
        index = self.cpu_index
        if isinstance(self.cpu_index, faiss.IndexPreTransform):
            print("oi")
            assert index.chain.size() == 1
            xq = self.cpu_index.chain.at(0).apply(xq)
            index = faiss.downcast_index(index.index)

        #Número de GPUS
        ngpu = index.count()
        print(ngpu)
        #Descobrir o que a função Downcast realiza.
        sub_index_0 = faiss.downcast_index(index.at(0))
        nprobe = sub_index_0.nprobe
        print(nprobe)

        #Inicializa as arrays que conterão as distâncias e os índices calculados em cada shard
        Dall = np.empty((ngpu, nq, k), dtype='float32')
        Iall = np.empty((ngpu, nq, k), dtype='int64')
        
        # batch_size
        bs = self.bs

        if bs <= 0:

            #Faz a busca padrão
            Dq, Iq = self.quantizer.search(xq, nprobe)

            #Define a função que irá fazer as buscas em shards
            def do_search(rank):
                gpu_index = faiss.downcast_index(index.at(rank))
                Dall[rank], Iall[rank] = gpu_index.search_preassigned(
                    xq, k, Iq, Dq)
            list(self.pool.map(do_search, range(ngpu)))
        else:
            #Provavelmente é uma queue de threads
            qq_pool = self.q_pool
            bs = self.bs

            #Função que faz a search pelo quantizer
            def coarse_quant(i0):
                if i0 >= nq:
                    return None
                return self.quantizer.search(xq[i0:i0 + bs], nprobe)

            #Função que faz a busca de fato
            def do_search(rank, i0, qq):
                gpu_index = faiss.downcast_index(index.at(rank))
                Dq, Iq = qq
                print(faiss.downcast_index(gpu_index))
                Dall[rank, i0:i0 + bs], Iall[rank, i0:i0 + bs] = gpu_index.search_preassigned(xq[i0:i0 + bs], k, Iq, Dq)

            qq = coarse_quant(0)

            #Aqui se realiza o sharding
            for i0 in range(0, nq, bs):
                qq_next = qq_pool.apply_async(coarse_quant, (i0 + bs, ))
                list(self.pool.map(
                    lambda rank: do_search(rank, i0, qq),
                    range(ngpu)
                ))
                qq = qq_next.get()

        #Retorna os kNN com o merge realizado
        return faiss.merge_knn_results(Dall, Iall)

def main():

    print("Faiss nb GPUs:", faiss.get_num_gpus())


    print("setting nb openmp threads to", 16)
    faiss.omp_set_num_threads(16)



    quantizer = faiss.IndexFlatL2(32)

    index_single = faiss.IndexIVFFlat(quantizer,32,4096,faiss.METRIC_L2)


    co = faiss.GpuMultipleClonerOptions()
    co.useFloat16 = True
    co.shard = False
    co.usePrecomputed = True
    co.shard_type = 1
    co.common_ivf_quantizer = True

    #Cria o índice
    print(f"move index to {faiss.get_num_gpus()} GPU")
    gpus = list(range(faiss.get_num_gpus()))
    res = [faiss.StandardGpuResources() for _ in gpus]
    index = faiss.index_cpu_to_gpu_multiple_py(res, index_single, co, gpus)




    #Treina o índice
    centroids = np.random.random((10000, 32)).astype('float32')
    quantizer = index_single

    xb = np.random.random((100000, 32)).astype('float32')


    quantizer.train(xb)
    quantizer.add(xb)

    #co.shard = False #Isso faz sentido?

    quantizer = faiss.index_cpu_to_gpu_multiple_py(res,quantizer,co,gpus)

    print("Index movido para as GPUS")
    batch_size = 320000

    print("Creating object")
    index = ShardedGPUIndex(quantizer,index,batch_size)

    print("Showing index attributes")
    print(vars(index))

    print("Creating numpy array")
    xb = np.random.random((1000000, 32)).astype('float32')

    print("Setting parameters")
    
    set_index_parameter(index,"quantizer_nprobe" , 20)
    

    print("BUSCA")
    D, I = index.search(xb, 100)

# https://github.com/facebookresearch/faiss/blob/main/benchs/bench_hybrid_cpu_gpu.py

if __name__ == "__main__":
    main()



import subprocess

yadiskLink = "https://yadi.sk/d/11eDCm7Dsn9GA"

# download base files
for i in range(37):
    command = 'curl ' + '"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
            + yadiskLink + '&path=/base/base_' + str(i).zfill(2) + '"'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = process.communicate()
    wgetLink = out.split(b',')[0][8:]
    wgetCommand = b'wget ' + wgetLink + b' -O base_' + bytes(str(i).zfill(2),'utf-8')
    print( "Downloading base chunk " + str(i).zfill(2) + ' ...')
    process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
    process.stdin.write('e')
    process.wait()

# download learn files
for i in range(14):
    command = 'curl ' + '"https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key=' \
            + yadiskLink + '&path=/learn/learn_' + str(i).zfill(2) + '"'
    process = subprocess.Popen(command, stdout=subprocess.PIPE, shell=True)
    (out, err) = process.communicate()
    wgetLink = out.split(',')[0][8:]
    wgetCommand = 'wget ' + wgetLink + ' -O learn_' + str(i).zfill(2)
    print( "Downloading learn chunk " + str(i).zfill(2) + ' ...')
    process = subprocess.Popen(wgetCommand, stdin=subprocess.PIPE, shell=True)
    process.stdin.write('e')
    process.wait()