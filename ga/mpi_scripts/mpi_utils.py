import numpy as np


def allocate_recvbuf(config, comm):

    comm_size = comm.Get_size()

    n_orgsnisms_per_process = config['runtime']['n_organisms'] // comm_size
    config['runtime']['n_orgsnisms_per_process'] = n_orgsnisms_per_process

    genes_size = sum(map(len, config['runtime']['genes_dict'].values()))
    config['runtime']['genes_size'] = genes_size

    recvbuf_genes = np.empty([comm_size, n_orgsnisms_per_process * genes_size])
    recvbuf_loss = np.empty([comm_size, n_orgsnisms_per_process * 1])
    recvbuf_status = np.empty([comm_size, n_orgsnisms_per_process * 1])

    recvbuf_dict = dict(genes=recvbuf_genes,
                        loss=recvbuf_loss,
                        status=recvbuf_status)

    return recvbuf_dict


def allgather(batch, recvbuf_dict, comm):

    sendbuf_genes = np.concatenate([sol.x for sol in batch])
    sendbuf_loss = np.array([sol.y for sol in batch])
    sendbuf_status = np.array([sol.status for sol in batch]).astype(float)

    comm.Allgatherv(sendbuf_genes,  recvbuf_dict['genes'])
    comm.Allgatherv(sendbuf_loss,   recvbuf_dict['loss'])
    comm.Allgatherv(sendbuf_status, recvbuf_dict['status'])


def population_from_recvbuf(recvbuf_dict, SolModel, config):

    recvbuf_genes = recvbuf_dict['genes']
    recvbuf_loss = recvbuf_dict['loss']
    recvbuf_status = recvbuf_dict['status']

    recvbuf_genes = recvbuf_genes.reshape((config['runtime']['n_organisms'], config['runtime']['genes_size']))
    recvbuf_loss = recvbuf_loss.flatten()
    recvbuf_status = recvbuf_status.flatten()

    population = []

    for i in range(config['runtime']['n_organisms']):
        sol = SolModel(recvbuf_genes[i].copy())
        sol._y = recvbuf_loss[i].copy()
        sol._status = recvbuf_status[i].copy()
        population.append(sol)

    return population
