# -*- coding: utf-8 -*-
"""
structural and dynamical analysis of the mysql network
"""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import EoN
import pickle
import matplotlib.ticker
from scipy.optimize import curve_fit
import time
  
# Text file data converted to integer data type
path = 'C:/nikyk/.Mallorca - UIB/Complex Networks/My Network - 13/'
path_figs = 'C:/nikyk/.Mallorca - UIB/Complex Networks/My Network - 13/figures/'
path_data = 'C:/nikyk/.Mallorca - UIB/Complex Networks/My Network - 13/data/'

data = np.loadtxt(path+"MySQL.txt", dtype=int)
edges = data[:,:2]

# initialise graph
G = nx.Graph()
G.add_edges_from(edges)


def structural_analysis(G, centrality, plot_args):
    """
    Take a networkx graph and calculate the following measures:
        - number of nodes
        - number of edges
        - density
        - S-W clustering coefficient 
        - Newman's transitivity index
        - avg shortest path length
        - diameter
        - degree assortativity
        - spectral bipartivity

    As well as:
        - histogram of degree distribution
        - pdf and cdf plots (optional)
        - top 25 nodes based on the following centrality measures (optional):
            - degree
            - closeness
            - betweenness
            - eigenvector
            - katz
            - page rank
            - subgraph
        

    Parameters
    ----------
    G : networkx graph
        the graph to be analysed.
    centrality : boolean
        whether to calculate centrality measures
    plot_args : list
        the parameters for plotting the pdf and the cdf of the degrees of the graph:
            - plot : boolean
                whwther to plot the pdf and the cdf. if false, the other parameters don't need to be specified.
            - name_graph : string
                the name of the graph, will be in the title
            - path_figs : string
                the path where to save the plots
            - log_scale : boolean
                whether to make the plots in log-log scale. This should be true for the actual and the albert-
                barabasi graph and false for the erdos-renyi graph.
                The default is True.

    Returns
    -------
    props : list
        a list with the following measures of the graph:
            - num_nodes
            - num_edges
            - density
            - S-W clustering coefficient 
            - Newman's transitivity index
            - avg shortest path length
            - diameter
            - degree assortativity
            - spectral bipartivity
    hist : numpy array
        histogram of the degree distribution.
    centrs : numpy array
        array with top 25 nodes acc to all centrality measures. if centrality=False, this will be a nan

    """
        
    assert nx.is_directed(G) == False
    if (nx.is_connected(G) == False):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    
    # number of nodes and edges
    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()
    print('Number of nodes: {}, number of edges: {}'.format(num_nodes, num_edges))
    
    # density
    d = nx.density(G)
    print('Density: {:5f}'.format(d))
    
    # s-w avg clustering coeff
    av_c = nx.average_clustering(G)
    print('Av. clust. coeff. W-S: {:5f}'.format(av_c))
    
    # GLOBAL clustering coeff (transitivity)
    tr = nx.transitivity(G)
    print('Global clust. coeff. (Newman transitivity): {:5f}'.format(tr))
    
    # av shortest path length
    av_l = nx.average_shortest_path_length(G)
    print('Av. shortest path length: {}'.format(int(av_l)))
    
    # diameter
    diam = nx.diameter(G)
    print('Diameter: {}'.format(diam))
    
    # ---- calc dist
    hist = np.bincount([d for n, d in G.degree()])
    
    if plot_args[0]:
        plot_deg_dist_cdf(hist, *plot_args[1:])
    
    # -- centrality
    if centrality:
        
        centrs = np.array(list(nx.degree_centrality(G).items()))
        centrs = np.flip(centrs[centrs[:, 1].argsort()][-25:,0].astype(int)).reshape(-1,1)
        
        for fun in [nx.closeness_centrality, nx.betweenness_centrality,
                    nx.eigenvector_centrality, nx.katz_centrality_numpy, nx.pagerank, 
                    nx.subgraph_centrality]:
            
            c = np.array(list(fun(G).items()))
            c = np.flip(c[c[:, 1].argsort()][-25:,0].astype(int)).reshape(-1,1)
            
            centrs = np.append(centrs, c, axis = 1)
        
    else:
        centrs = np.nan
    
    # - degree assortativity
    dass = nx.degree_pearson_correlation_coefficient(G)
    print('degree assortativity:{:.5f}'.format(dass))
    
    # -- spectral bipartivity
    bp = nx.algorithms.bipartite.spectral_bipartivity(G)
    print('spectral bipartivity:{:.5f}'.format(bp))
    
    props = np.array([num_nodes, num_edges, d, av_c, tr, av_l, diam, dass, bp])

    return props, hist, centrs

def plot_deg_dist_cdf(data, name_graph, path, log_scale):
    """
    Plot the pdf and inverse cdf of the degree distribution.
    Saves a png and an eps figure

    Parameters
    ----------
    data : numpy array
        histogram of the distribution, not normalised.
    name_graph : string
        the name of the graph - will be in the title of the plots.
    path : string
        the path where to save the plots.
    log_scale : boolean, optional
        whether to make the plots in log-log scale. This should be true for the actual and the albert-
        barabasi graph and false for the erdos-renyi graph.
        The default is True.

    Returns
    -------
    icdf : numpy array
        the inverse cdf, normalised
    icdf_unnorm : numpy array
        the inverse cdf, unnormalised

    """
     
    """ Plot Distribution """
   
    # normalised
    plt.plot(range(len(data)),data/np.sum(data),'bo',markersize=3, alpha = 0.3)
    
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.gca().get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.gca().get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        
    plt.ylabel('Freq')
    plt.xlabel('Degree')
    plt.savefig(path + name_graph + '_pdf.eps')
    plt.savefig(path + name_graph + '_pdf.png')
    plt.show()
    plt.clf()
 
    """ Plot CCDF """
    s = float(data.sum())
    cdf = data.cumsum(0)/s
    icdf = 1-cdf
    icdf_unnorm = s - data.cumsum(0)
    plt.plot(range(len(cdf)),icdf,'bo', markersize=3, alpha = 0.3)
    
    if log_scale:
        plt.yscale('log')
        plt.xscale('log')
        plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.gca().get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
        plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
        plt.gca().get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
     
    plt.ylabel('CCDF')
    plt.xlabel('Degree')

    plt.savefig(path + name_graph + '_cdf.eps')
    plt.savefig(path + name_graph + '_cdf.png',dpi=1000)
    plt.show()
    plt.clf()
    
    return icdf, icdf_unnorm

def save_net_data(name_graph, path, dict_obj):
    
    for name_obj, obj in dict_obj.items():
        
        fname = '{}_{}.txt'.format(name_graph, name_obj)
        np.savetxt(path+fname, obj, fmt='%.5e', delimiter=' ', newline='\n')



# ----------------------------------------------Actual network

plot_args = [False]
net_props, hist, centrs = structural_analysis(G, centrality=True, plot_args=plot_args)

name_graph = 'MySQL'
log_scale = True
icdf_norm, icdf_unnorm = plot_deg_dist_cdf(hist, name_graph, path_figs, log_scale)
dist = np.insert(hist.reshape(-1,1), 0, np.arange(len(hist)), axis=1).astype('float64')
dist = np.insert(dist, 2, icdf_norm, axis=1)

# save data
dic = {}
dic['measures'] = net_props
dic['nodes-dist'] = dist
# dic['central-nodes'] = centrs
save_net_data(name_graph, path_data, dic)

# save centers as latex table
np.savetxt(path_data+'myqsl_centers.txt',centrs, delimiter=' & ', fmt='%i', newline=' \\\\\n\\hline\n')

for i in np.unique(centrs):
    
    n = np.count_nonzero(centrs == i)
    if n > 3:
        print('Num:{},n:{}'.format(i,n))
        
        
        

# --------------------- fit dist


xs = np.arange(len(icdf_unnorm))

# cut offs for where the line is straigh to fit the exponent
cut_off_min = 1
cut_off_max = 500
icdf_cut = icdf_unnorm[np.where((icdf_unnorm > cut_off_min) & (icdf_unnorm < cut_off_max))]
xs_cut = xs[np.where((icdf_unnorm > cut_off_min) & (icdf_unnorm < cut_off_max))]

def myExpFunc(x, a, b):
    
    return a * np.power(x, b)

popt, pcov = curve_fit(myExpFunc, xs_cut, icdf_cut)
gamma = -popt[1]
fitted = myExpFunc(xs_cut, popt[0], popt[1])
pfitted = np.pad(fitted, (np.where(icdf_unnorm >= cut_off_max)[0].shape[0], 
                          np.where(icdf_unnorm <= cut_off_min)[0].shape[0]), constant_values =
                  np.nan)
# plot
plt.plot(xs, icdf_unnorm, 'bo', markersize=3, alpha = 0.3)
plt.plot(pfitted, 'k', label=r'$\gamma={:.2f}$'.format(gamma+1))
plt.xscale('log')
plt.yscale('log')
plt.ylabel('CCDF')
plt.xlabel('Degree')

plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
plt.gca().get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())

plt.legend()
plt.savefig(path_figs + name_graph + '_fit-cdf.eps')
plt.savefig(path_figs + name_graph + '_fit-cdf.png',dpi=1000)
plt.show()
plt.clf()



# -- visualise network 
fig, ax = plt.subplots()
pos_kam = nx.kamada_kawai_layout(G)
nx.draw(G, node_size = 30, alpha = 0.6, pos = pos_kam, ax = ax)
plt.savefig(path_figs+'the mysql net.png', format='png',dpi=1000)


# -------------------------------------------- ER and AB networks

    
seed = 123
rgn = np.random.RandomState(seed)
max_degree = hist.shape[0]
num_nodes = int(net_props[0])
num_edges = int(net_props[1])

num_nets = 100
num_props = 9


# -- ER

p = 2*num_edges/(num_nodes*(num_nodes-1)) # probability for the creation of the graph

er_nets = np.zeros((num_nets,num_props))
er_nets_ddists = np.zeros((num_nets,max_degree))

for i in range(num_nets):
    ER = nx.fast_gnp_random_graph(num_nodes, p, seed = rgn)
    er_nets[i,:], hist, _ = structural_analysis(ER, centrality = False, plot_args = [False])
    # pad hist in case the max degree is different for each realisation of the graph
    pad_num = max_degree-len(hist)
    er_nets_ddists[i,:] = np.pad(hist, (0, pad_num), 'constant')

# -- calc mean and std for each stat
er_nets_means = np.mean(er_nets,axis = 0)
er_nets_avs = np.hstack([er_nets_means.reshape(-1,1), np.std(er_nets,axis=0).reshape(-1,1)])

# - for dists
er_nets_meandist = np.mean(er_nets_ddists,axis=0).astype(int)
er_nets_stds = np.std(er_nets_ddists,axis=0)

icdf_norm, _ = plot_deg_dist_cdf(er_nets_meandist, 'Erdos-Renyi-gnp-av100--log', path_figs, log_scale=True)
_,_ = plot_deg_dist_cdf(er_nets_meandist, 'Erdos-Renyi-gnp-av100-non-log', path_figs, log_scale=False)
er_nets_dists = np.hstack([
                          np.hstack([np.arange(len(er_nets_meandist)).reshape(-1,1), er_nets_meandist.reshape(-1,1)]),
                          np.hstack([er_nets_stds.reshape(-1,1), icdf_norm.reshape(-1,1)])
                          ])

# -- save data
dic = {}
dic['measures'] = er_nets_avs
dic['node-dists'] = er_nets_dists
name_graph = 'ER-gnp-av100'
save_net_data(name_graph, path_data, dic)


# - draw one graph

ER_lc = ER.subgraph(max(nx.connected_components(ER), key=len))
pos = nx.kamada_kawai_layout(ER_lc)
pos_sp = nx.spring_layout(ER_lc)
pos_c = nx.circular_layout(ER_lc, scale=0.5)
nx.draw(ER_lc, node_size = 50, alpha = 0.6, pos=pos_c)


# ------------------------------------------------- albert-barabasi network

# -- AB

ab_nets = np.zeros((num_nets,num_props))

for i in range(num_nets):

    m = int(np.round(num_edges/num_nodes,0))
    AB = nx.generators.random_graphs.barabasi_albert_graph(n=num_nodes, m=m, seed = rgn)
    ab_nets[i,:], hist, _ = structural_analysis(AB, centrality = False, plot_args = [False])
    
    if i == 0: # here the hists can be of diff length and we are taking avgs so we need to make sure they are all of
                # equal length
        max_degree = len(hist)
        ab_nets_ddists = np.zeros((num_nets,max_degree))
        
    # pad hist in case the max degree is more than the current or if it is less, cut the current
    pad_num = max_degree-len(hist)
    if (pad_num < 0):
        hist = hist[:max_degree]
    elif (pad_num > 0):
        hist = np.pad(hist, (0, pad_num), 'constant')
        
    ab_nets_ddists[i,:] = hist

# -- calc mean and std for each stat
ab_nets_means = np.mean(ab_nets,axis = 0)
ab_nets_avs = np.hstack([ab_nets_means.reshape(-1,1), np.std(ab_nets,axis=0).reshape(-1,1)])

# - for dists
ab_nets_meandist = np.mean(ab_nets_ddists,axis=0).astype(int)
ab_nets_stds = np.std(ab_nets_ddists,axis=0)
icdf_norm,_ = plot_deg_dist_cdf(ab_nets_meandist, 'Albert-Barabasi-2', path_figs, log_scale = True)
ab_nets_dists = np.hstack([
                          np.hstack([np.arange(len(ab_nets_meandist)).reshape(-1,1), ab_nets_meandist.reshape(-1,1)]),
                          np.hstack([ab_nets_stds.reshape(-1,1), icdf_norm.reshape(-1,1)])
                          ])

# -- save data
dic = {}
dic['measures'] = ab_nets_avs
dic['node-dists'] = ab_nets_dists
name_graph = 'AB-2'
save_net_data(name_graph, path_data, dic)



# - draw one graph

pos = nx.kamada_kawai_layout(AB)
pos_sp = nx.spring_layout(AB)
pos_c = nx.circular_layout(AB, scale=0.5)
nx.draw(AB, node_size = 50, alpha = 0.6, pos=pos_c)



# --------- plot dists on one axis

name_graph = 'pdf_comp-2'
fig, ax = plt.subplots()
ax.plot(dist[:,0],dist[:,1]/np.sum(dist[:,1]),'bo',markersize=3, alpha = 0.6, label = 'MySQL')
ax.plot(er_nets_dists[:,1]/np.sum(er_nets_dists[:,1]),'go',markersize=3, alpha = 0.6, label = 'E-R')
ax.plot(ab_nets_dists[:,1]/np.sum(ab_nets_dists[:,1]),'ro',markersize=3, alpha = 0.6, label = 'A-B')

if log_scale:
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    
plt.ylabel('PDF')
plt.xlabel('Degree')
# plt.title(name_graph)
plt.legend()
plt.savefig(path_figs + name_graph + '.eps')
plt.savefig(path_figs + name_graph + '.png',dpi=1000)
plt.show()
plt.clf()

name_graph = 'ccdf_comp-2'
fig, ax = plt.subplots()
ax.plot(range(len(dist[:,2])),dist[:,2],'bo',markersize=3, alpha = 0.6, label = 'MySQL')
ax.plot(er_nets_dists[:,3],'go',markersize=3, alpha = 0.6, label = 'E-R')
ax.plot(ab_nets_dists[:,3],'ro',markersize=3, alpha = 0.6, label = 'A-B')

if log_scale:
    plt.yscale('log')
    plt.xscale('log')
    plt.gca().get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().get_xaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    plt.gca().get_yaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    plt.gca().get_yaxis().set_minor_formatter(matplotlib.ticker.NullFormatter())
    
plt.ylabel('CCDF')
plt.xlabel('Degree')
# plt.title(name_graph)
plt.legend()
plt.savefig(path_figs + name_graph + '.eps')
plt.savefig(path_figs + name_graph + '.png',dpi=1000)
plt.show()
plt.clf()


# ----------------------------------------------------- communities

comm = nx.algorithms.community.centrality.girvan_newman(G)

for comm in comm:
    print(comm)
    
    

# ------------------------------------------------ SIMULATE SIS/SIR

# tmax = 10**5
iterations = 10  #run 5 simulations
tau = 0.6      #beta - transmission rate
gamma = 0.5    #gamma - recovery rate


for counter in range(iterations): #run simulations
    t, S, I, R = EoN.fast_SIR(G, tau, gamma)
    
    if counter == 0:
        plt.plot(t, I, color = 'k', alpha=0.3, label='Simulation')
    plt.plot(t, I, color = 'k', alpha=0.3)
plt.xlabel('$t$')
plt.ylabel('Number infected')
plt.legend()


# -----------------------------------  epidemic_diagram

sim = 'sis' #['sis','sim']
iterations = 10**3
dtau=0.1
gamma = 1

taus = np.arange(0,1,dtau)


infs_per_tau = np.zeros((len(taus),1))
for it, tau in enumerate(taus):
    print(tau)    
    t0 = time.time()

    infected = 0
    for counter in range(iterations): #run simulations
        
        if sim == 'sis':
            t, S, I = EoN.fast_SIS(G, tau, gamma)
        else:
            t, S, I, R = EoN.fast_SIR(G, tau, gamma)
            
        infected += I[-1] if sim == 'sis' else np.max(I)
        
    infected = infected/iterations
    infs_per_tau[it] = infected
    
    t1 = time.time()
    total = t1-t0
    print('Elapsed seconds: {:.2f}'.format(total))
        
plt.plot(taus, infs_per_tau, 'ko', markersize = 3, alpha=0.6,
         label = r'$\gamma={}$'.format(gamma))
plt.xlabel(r'$\tau$')
plt.ylabel('Number infected')
plt.title('SIS mySQL Network')
plt.legend()

sim_name = 'epid-diag_mySQL_{}_dtau={}_gamma={}_iters={}'.format(sim,
                                                                 dtau,
                                                                 gamma,
                                                                 iterations)
plt.savefig(path_figs+sim_name+'.png')
plt.show()


# -------- save results object
with open(path+sim_name+'.pkl', 'wb') as file:
      
    # A new file will be created
    pickle.dump(infs_per_tau, file)
    
# -------- read results object

with open(path+sim_name+'.pkl', 'rb') as file:
      
    # A new file will be created
    res_test = pickle.load(file)

