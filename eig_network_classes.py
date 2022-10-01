import copy as cp
import multiprocessing as mpr
import collections as clts
from brian2 import *

#-----------------------------------------------------------------------------------------------------------------------
# Utility to handle dictionaries
#-----------------------------------------------------------------------------------------------------------------------
def merge_dicts(dict_0,*nargs):
    """
    Merge arbitrary number of dictionaries with dict_0.
    WARNING: This will change the original dictionary dict_0 and pass it by reference.

    Input arguments:
    - dict_0   : First dictionary to merge
    - *nargs   : list of further dictionaries to merge

    Return merged dictionary (with unique keys).
    Each key contains the last presentation.

    Maurizio De Pitta', The University of Chicago, Apr 27, 2016.
    """
    if np.size(nargs)==0 : return dict_0
    if np.size(nargs)==1 :
        if type(dict_0)==type(clts.OrderedDict()):
            return clts.OrderedDict(dict_0,**nargs[0])
        else:
            return dict(dict_0, **nargs[0])
    return merge_dicts(dict_0,merge_dicts(nargs[0],*nargs[1:]))

#-----------------------------------------------------------------------------------------------------------------------
# Some standard lambdas
#-----------------------------------------------------------------------------------------------------------------------
u_ = lambda u0,xi,gs : u0 + (xi-u0)*gs
r_eff = lambda nu,trp : nu/(1.+nu*trp)
safe_threads = lambda n,safe_cpu : int(min(max(1,n),mpr.cpu_count()-1)) if safe_cpu else int(min(max(1,n),mpr.cpu_count()))  ## This will assure that the number of threads is 1<=n<=n_cpu-1
index_from_string = lambda string : int(''.join([i for i in list(string) if i.isdigit()]))    # Retrieve 'full' number from string: used in monitor handling for clustered networks

#-----------------------------------------------------------------------------------------------------------------------
# Model modules
#-----------------------------------------------------------------------------------------------------------------------
#-----------------------------------------------------------------------------------------------------------------------
# Neuron/cell modules
#-----------------------------------------------------------------------------------------------------------------------
def periodic_nodes(N,rates,name='p',dt=None):
    if isinstance(rates,input.timedarray.TimedArray):
        eqs = Equations('dv/dt = stimulus(t) : 1')
        nmspace = {'stimulus': rates}
    else:
        eqs = Equations('dv/dt = rate : 1')
        nmspace = None
    cells = NeuronGroup(N,eqs,
                        threshold='v>=1.0',
                        reset='v=0.0',
                        name=name,
                        namespace=nmspace,
                        dt=dt)
    if not isinstance(rates,str): cells.rate = rates
    cells.v = 0.0
    return cells

@check_units(trp=second)
def poisson_nodes(N,rates,trp=0.0*second,
                  name='p',
                  dt=None):
    # Create equations to also handle case of tme-variant stimulation
    if isinstance(rates,input.timedarray.TimedArray):
        eqs = Equations('rate = stimulus(t) : Hz')
        nmspace = {'stimulus': rates}
    else:
        eqs = Equations('rate : Hz')
        nmspace = None
    cells = NeuronGroup(N,eqs,
                          threshold='rand()<rate*dt',
                          refractory=trp,
                          name=name,
                          namespace=nmspace,
                          dt=dt)
    if not isinstance(rates,input.timedarray.TimedArray): cells.rate = rates
    return cells

def cell_nodes(N,params,
               name='e',
               noise=False,
               dynamic_reset=False,
               method=None,
               dt=None,
               add_parameters=None):

    """
    Generate Neuron/Astrocyte LIF models/network nodes.

    Input arguments:
    - N              : Scalar, Number of cells to generate
    - params         : Dictionary, Model parameters as a dictionary of values with units
    - name           : String, Name of the cell population in the Brain's Network Object
    - noise          : Boolean. If True add noise as a OU process to the LIF equations
    - dynamic_reset  : Boolean. This changes the after-spike reset from vr to v(+) = v(-) + vr - vt.
    - method         : String. Integration method.
    - dt             : Scalar with units. Time step of integration.
    - add_parameters : Additionally include parameters beyond those in the params dict when dealing with custom equations / linking

    Returns:
    - cells          : A NeuronGroup of LIF cells.
    """

    # This is necessary with the inclusion of TimeArray in the equations, since it avoids inconsistencies between
    # dt in the empy TimeArray definition, and the dt in the NeuronGroup, which would raise a red flag for the integration
    # method
    if dt is None: dt = defaultclock.dt

    # NOTE: Equations also include mu(t,i) and sigma(t,i) which by default are set to zero (not activated)
    if not noise:
        eqs = Equations('''
                        # LIF equation
                        dv/dt = (vl-v + ix + mu(t,i))/taum : 1 (unless refractory)
                        ix : 1
                        ''')
        nmspace = {'mu': TimedArray(np.zeros((2,N)),dt=dt)}
    else:
        eqs = Equations('''
                        # LIF equation
                        dv/dt = (vl-v + ix + mu(t,i))/taum + (sx + sigma(t,i))*xi/(taum**.5) : 1 (unless refractory)
                        ix : 1
                        sx : 1                        
                        ''')
        nmspace = {'mu': TimedArray(np.zeros((2,N)),dt=defaultclock.dt), 'sigma': TimedArray(np.zeros((2,N)),dt=dt)}
    if not method: method = 'euler'

    # Custom parameters
    if add_parameters!=None:
        for p in add_parameters:
            par_string = p[0] + ' : ' + str(p[1])
            eqs += Equations('''{par_string}'''.format(par_string=par_string))

    if dynamic_reset:
        reset = 'v+=vr-vt'
    else:
        reset = 'v=vr'

    cells = NeuronGroup(N,eqs,
                          threshold='v>=vt',
                          reset=reset,
                          refractory='trpn',
                          method=method,
                          dt=dt,
                          namespace=merge_dicts(params,nmspace),
                          name=name,
                          order=0)

    # Automatically initialize the value of noise parameters to those in the parameters
    # Except for the clustered network, noise parameters could be simply passed in the namespace
    # But clustering and stimulation of individual clusters requires ix and sx to be set as const variables
    if noise:
        cells.ix = params['ix']
        cells.sx = params['sx']

    return cells

#-----------------------------------------------------------------------------------------------------------------------
# Synaptic connections for standard N-N connections
#-----------------------------------------------------------------------------------------------------------------------
def probabilistic_synapses(neuron_source,neuron_target,
                           name='S_xx*',
                           dt=None,delay=None):
    model = '''w   : 1 (constant)
               r   : boolean
               u_0 : 1 (constant)
            '''
    eqs_pre = '''r = (rand()<u_0)
                 v_post += r*w
              '''
    synapses = Synapses(neuron_source,neuron_target,
                        model=model,on_pre=eqs_pre,
                        dt=dt,name=name,delay=delay)

    # Set a reset for r variable after the execution of the synapse code and any monitor (the latter usually recording at when='end' by default)
    synapses.run_regularly('''r=False''',when='resets')
    # To set after definition:
    # synapses.connect(...)
    # synapses.w = ...
    # synapses.u_0 = ...
    return synapses

#-------------------------------------------------------------------------------------------------
# Modules for Tripartite synapse connections
#-------------------------------------------------------------------------------------------------
def SynapticGroup(Nsyn,params,
                 gliot=False,
                 method=None,
                 dt=None,
                 name='syn_main*',
                 add_parameters=None):
    """
    This is the main object of the TripartiteConnection. It contains synaptic variables.
    The params require {'u0': ..., 'taup': ...}.
    Variables to be assigned after instantiation: w, alpha (only if gliot==True)
    alpha (xi) (if gliot); u_0 (gliot==False) or u0 (gliot==True); and synaptic weight w must be assigned accordingly.

    Input arguments:
    - Nsyn   : Number of synapses
    - params : Model parameters as a dictionary with units
    - gliot  : Boolean  If True make synapse equations include modulation of synaptic release by gliotransmission
    - method : String for integration method (Default: 'euler')
    - dt     : Time step for integration (with units)
    - name   : Synapse name (within Brian Network's Object)
    - add_parameters : Additionally include parameters beyond those in the params dict when dealing with custom equations / linking

    Return:
    - synapses : A Brian NeuronObject coding for synapses.

    NOTE: This way of creating synapses as NeuronObjects is exclusively used by the TripartiteConnection method. All
    synapses in the network should be generated by TripartiteConnections, regarldess.
    """
    # Parameters must contain 'taup' and 'u0'
    eqs = Equations('''
                    r : boolean
                    from_neuron : integer (constant)
                    to_neuron   : integer (constant)
                    from_glia   : integer (constant)
                    to_glia     : integer (constant)
                    ''')

    if gliot:
        eqs += Equations('''dgamma_S/dt = -gamma_S/taup   : 1
                            u_0 = u0 + (alpha-u0)*gamma_S : 1
                            u0                            : 1 (constant)
                            alpha                         : 1 (constant)
                         ''')
    else:
        eqs += Equations('''u_0 : 1''')

    # WARNING: Following the above you need to initilize u_0 in the gliot==False, but u0 in the gliot=True scenario

    # Custom parameters
    if add_parameters!=None:
        for p in add_parameters:
            par_string = p[0] + ' : ' + str(p[1])
            eqs += Equations('''{par_string}'''.format(par_string=par_string))

    if not method: method = 'euler'
    synapses = NeuronGroup(Nsyn,eqs,
                          threshold='r',
                          reset='r=False',
                          method=method,
                          dt=dt,
                          namespace=params,
                          name=name,
                          order=0)
    return synapses

def gliot_pathways(astro_source, syn_target, params,
                   name='gliot*',
                   method=None,
                   dt=None,
                   delay=None):
    """
    Model pathway of gliotransmission. Requires a SynapticGroup as syn_target. Connections must be specified outside
    of this method, after call. Instantation of wsg also requires assignment after connections are established.

    Input arguments:
    - astro_source : The Astrocyte NeuronGroup that is the source of gliotransmission
    - syn_target   : The Synapses that are targeted by gliotransmission
    - params       : Dictionary of model parameters with values
    - name         : Gliot Pathway name (within Brian Network's Object)
    - wsg          : Heterotypic connection weights
    - dt           : Simulation time step (with units)
    - delay        : Delay in gliotransmission (with units)

    Return:
    - gliot        : A Synapse Object characterizing a Gliotransmission Pathway
    """

    # Glia-to-syn connections are provided as variables (constants)
    eqs = Equations('''wsg : 1 (constant)''')

    if params['taug'] >= 1.0*ms:
        # Release subjected to depletion
        eqs += Equations('''
              dx_A/dt = (1-x_A)/taug : 1 (event-driven)
              r_A : 1
              ''')
        eqs_pre = '''
                  r_A = (rand()<ua)
                  x_A -= r_A*x_A
                  gamma_S_post += r_A*wsg*(1-gamma_S_post)
                  '''
    else:
        # Instantaneous fixed release
        eqs += Equations('''
              r_A : 1
              ''')
        eqs_pre = '''
                  r_A = (rand()<ua)
                  gamma_S_post += r_A*wsg*(1-gamma_S_post)
                  '''

    # Instantiate class
    gliot = Synapses(astro_source,syn_target,eqs,
                     on_pre=eqs_pre,
                     namespace=params,
                     method=method,
                     name=name,
                     dt=dt,
                     delay=delay,
                     order=0)
    return gliot

def probabilistic_connections(neuron_source,syngroup_target,name='syn_prob*'):
    """
    Probabilistic connection used by shared synapses

    Input arguments:
    - source  : Source NeuronGroup
    - target  : Target NeuronGroup
    - name    : Synapse name in Brian

    Return:
    - prob_connection : Synapses Object
    """
    on_pre = '''r_post = (rand()<u_0_post)'''
    prob_connection = Synapses(neuron_source,syngroup_target,on_pre=on_pre,name=name,order=0)
    return prob_connection

def synaptic_connection(syngroup_source,neuron_target,
                        name='syn_conn*',
                        delay=None):
    """
    Simple relay of presynaptic activity where the presynaptic source is a SynapticGroup.

    Input arguments:
    - source : Synapse Group source
    - target : Neuron Group target
    - name   : synaptic connection name in Brain
    - delay  : Dealy in synaptic transmission (with units)

    Return:
    - synapses : A Synapes Brain object

    NOTE: The user must set connections AND 'w' variable after using this method.
    """
    on_pre = 'v_post += r_pre*w'
    synapses = Synapses(syngroup_source,neuron_target,
                        model='w : 1',on_pre=on_pre,
                        name=name,delay=delay,order=10)
    return synapses

def TripartiteConnection(Pre,Post,Glia,_N_syn,syn_pars,
                         _from_neuron_indexes,_to_neuron_indexes,
                         _from_glia_indexes,_to_glia_indexes,
                         _target_syn_indexes,_source_syn_indexes,
                         jxx=0.,wxx=0.,wsg=0.,
                         shared=True,gliot=False,gliot_pars=None,
                         method=None,dt=None,name='tri'):
    """
    Input arguments:
    (NOTE: '*' denotes input arguments that can be mutable objects (e.g. dictionary entries), but it is guaranteed that they will not be changed)
    - Pre: Presynaptic Neuron Group
    - Post: Postsynaptic Neuron Group
    - Glia: Glia cell group
    -* _N_syn: Number of total synaptic connections between Pre and Post group
    - syn_pars  : {'u0': ..., 'taup': ...} Dictionary of synaptic parameters (including presynaptic receptors)
    -* _from_neuron_indexes: Presynaptic neuron indexes for synaptic connections (i.e. edges['xx'][0])
    -* _to_neuron_indexes: Postsynaptic neurons indexes for synaptic connections (i.e. edges['xx'][1] )
    -* _from_glia_indexes: Indexes of glial cells that modulate synapses (i.e. edges['xxg'][0])
    -* _to_glia_indexes: Indexes of glial cells targeted by the synapses synapses (i.e. edges['gxx'][1])
    -* _target_syn_indexes: Indexes of synapses targeted by glial cells (i.e. edges['xxg'][1])
    -* _source_syn_indexes: Indexes of synapses stimulating glial cells (i.e. edges['gxx'][0])
    - jxx: synaptic weights
    - wxx: synapse-to-glial weight
    - wsg: glia-to-synapse weight
    - shared: Boolean   Whether synapses are going to stimulate glia
    - gliot: Boolean   Whether synapses are going to be modulated by glia
    - gliot_pars: {'taug': ..., 'ua': ..., 'alpha': ..., 'ICpre': ....} Dictionary of gliotransmitter parameters
    - method: Integration method for SynapticGroup and gliot_pathways
    - dt: time step for integration
    - name: Name of the module -- will create a series of children also depending on the configuration

    Return:
    - network object(s) list
    """

    # Generate copies of the indexes to avoid overwriting of mutable arguments
    from_neuron_indexes = cp.deepcopy(_from_neuron_indexes)
    to_neuron_indexes = cp.deepcopy(_to_neuron_indexes)

    # This group returns multiple objects, whose number and nature depends on the shared/gliot arguments
    # IMPORTANT: This group implements already all the connections, but does not initialize variables!!
    if (not shared) and (not gliot):
        # This is the standard (non-tripartite) connection to run the default EI case with probabilistic synapses
        S_main = probabilistic_synapses(Pre,Post,name=name,dt=dt)
        S_main.connect(i=from_neuron_indexes,j=to_neuron_indexes)
        S_main.u_0 = syn_pars['u0']
        S_main.w = jxx
        # Return Tuple of size 1
        return S_main,
    else:
        # -----------------------------------------------------------------------------
        # Pre-processing to also separate tripartite from non-tripartite synapses
        # -----------------------------------------------------------------------------
        # Preliminary checks
        assert np.size(_source_syn_indexes)==np.size(_to_glia_indexes),'Source Synaptic Indexes must be of same length of To Glia Indexes for group'+name
        assert np.size(_target_syn_indexes)==np.size(_from_glia_indexes),'Target Synaptic Indexes must be of same length of From Glia Indexes for group'+name
        # The following is a strict check that warns on the limits of the current implementation (to drop in the future, when you can differentiate between synaptic groups that are shared and have gliot independently)
        if shared and gliot:
            if np.size(_to_glia_indexes)!=np.size(_from_glia_indexes):
                print('WARNINNG: LIMITED IMPLEMENTATION: Different to/from glial indexes are currently allowed only for gliot subsets of shared synapses')

        # Generate copies of the indexes to avoid overwriting of mutable arguments
        N_syn = cp.deepcopy(_N_syn)
        # WARNING: This will change the dictonary
        if shared:
            to_glia_indexes = cp.deepcopy(_to_glia_indexes)
        if gliot:
            from_glia_indexes = cp.deepcopy(_from_glia_indexes)

        # Verify whether there are synapses that are not tripartite -- such as in the case of confined clustered networks or p<1.0
        # for tripartite synapses
        # If no-tripartite synapses are detectde then update N_syn, to/from neuron_indexes and set a flag "no_tripartite" to True to generate at the end an extra synaptic
        # group of simple probabilistic connections
        if (shared and N_syn!=np.size(_source_syn_indexes))or(gliot and N_syn!=np.size(_target_syn_indexes)):
            no_tripartite = True
            if not (shared and gliot and np.size(_source_syn_indexes)!=np.size(_target_syn_indexes)):
                no_partial = True # No partial synapses detected
                # This is the case used so far in all simulations with 'clustered-within'
                tripartite_indexes = np.union1d(_source_syn_indexes,_target_syn_indexes).astype(np.int32)
                from_neuron_indexes = from_neuron_indexes[tripartite_indexes]
                to_neuron_indexes = to_neuron_indexes[tripartite_indexes]
                # Need also to rearrange the from/to glia indexes according to the permutations by union1d which is rearranging in sorted order
                if gliot:
                    from_glia_indexes = from_glia_indexes[np.argsort(_target_syn_indexes)]
                if shared:
                    to_glia_indexes = to_glia_indexes[np.argsort(_source_syn_indexes)]
                N_nog = N_syn-np.size(tripartite_indexes)   # Number of non-tripartite synapses
                N_syn = np.size(tripartite_indexes)         # Number of tripartite synapses
            else:
                if shared and gliot:
                    no_partial = False # There are partial synapses in this case
                    # This handles the special case where you have some synapses that are only stimulating/modulated by glia
                    # TODO: This can be extended to handle the general case of any configuration: currently ONLY handles the case where gliot synapses are a subset of shared synapses
                    # TODO: This has been tested only in the special case of 2-cluster network with p_eeg = p_ieg = [0.0,1.0]
                    tripartite_indexes,fs_indexes,ts_indexes = np.intersect1d(_source_syn_indexes, _target_syn_indexes, return_indices=True)
                    # Indexes of effective tripartite synapses
                    from_neuron_indexes = from_neuron_indexes[tripartite_indexes]
                    to_neuron_indexes = to_neuron_indexes[tripartite_indexes]
                    from_glia_indexes = from_glia_indexes[ts_indexes]
                    to_glia_indexes = to_glia_indexes[fs_indexes]
                    # Generate Synapse numbers
                    N_tri = np.size(tripartite_indexes) # Tripartite synases
                    N_shr = np.size(np.setdiff1d(_source_syn_indexes,tripartite_indexes)) # Shared-only synapses
                    N_nog = N_syn - N_tri - N_shr       # Number of non-tripartite synapses
                    N_syn = N_tri
        else:
            no_tripartite = False
            no_partial = True

        # -----------------------------------------------------------------------------
        # Building modules
        # -----------------------------------------------------------------------------
        # Generate Main Synaptic group
        S_main = SynapticGroup(N_syn,syn_pars,gliot=gliot,method=method,dt=dt,name=name)
        S_main.from_neuron = from_neuron_indexes
        S_main.to_neuron = to_neuron_indexes

        # Generate probabilistic connections
        S_prob = probabilistic_connections(Pre,S_main,name=name+'_prob')
        S_prob.connect('i==from_neuron_post')

        # Generate relay connections
        to_post = synaptic_connection(S_main,Post,name=name+'_n')
        to_post.connect('j==to_neuron_pre')
        to_post.w = jxx

        if shared:
            S_main.to_glia = to_glia_indexes
            to_glia = synaptic_connection(S_main,Glia,name=name+'_g')
            to_glia.connect('j==to_glia_pre')
            to_glia.w = wxx

        if gliot:
            S_main.from_glia = from_glia_indexes
            from_glia = gliot_pathways(Glia,S_main,gliot_pars,method=method,dt=dt,name='g_'+name)
            from_glia.connect('i==from_glia_post')
            from_glia.wsg = wsg
            # In this case we also want to allocate ICs properly setting gamma_S
            S_main.gamma_S = gliot_pars['ICpre']
            # And set the 'u0' and 'alpha' parameters for gliotransmission
            S_main.u0 = syn_pars['u0']
            S_main.alpha = gliot_pars['alpha']
        else:
            S_main.u_0 = syn_pars['u0']

        # Handling of output
        # NOTE: the simple handling by extending a list cannot be used in standalone as the Synaptic objects are not allocated till build is invoked
        if shared and gliot:
            network_objects = [S_main,S_prob,to_post,to_glia,from_glia]
        else:
            if shared:
                network_objects = [S_main, S_prob, to_post, to_glia]
            if gliot:
                network_objects = [S_main, S_prob, to_post, from_glia]

        if (not no_partial) and (N_shr>0):
            S_shr = SynapticGroup(N_shr,syn_pars,gliot=False,method=method,dt=dt,name=name+'_shr')
            shared_indexes = np.setdiff1d(_source_syn_indexes,tripartite_indexes)
            S_shr.from_neuron = _from_neuron_indexes[shared_indexes]
            S_shr.to_neuron = _to_neuron_indexes[shared_indexes]
            S_shr.u_0 = syn_pars['u0']
            # Generate probabilistic connections
            S_prob_shr = probabilistic_connections(Pre,S_shr,name=name+'_prob_shr')
            S_prob_shr.connect('i==from_neuron_post')
            # Generate relay connections
            to_post_shr = synaptic_connection(S_shr,Post,name=name+'_n_shr')
            to_post_shr.connect('j==to_neuron_pre')
            to_post_shr.w = jxx
            # Add connections to glia
            S_shr.to_glia = _to_glia_indexes[np.isin(_source_syn_indexes,shared_indexes)]
            to_glia_shr = synaptic_connection(S_shr,Glia,name=name+'_g_shr')
            to_glia_shr.connect('j==to_glia_pre')
            to_glia_shr.w = wxx
            network_objects.extend([S_shr,S_prob_shr,to_post_shr,to_glia_shr])

        if no_tripartite and (N_nog>0):
            # Generate non-tripartite synapses
            nog_indexes = np.setdiff1d(np.arange(np.size(_from_neuron_indexes)),np.union1d(tripartite_indexes,_source_syn_indexes))
            S_not = probabilistic_synapses(Pre, Post, name=name+'_nog', dt=dt)
            S_not.connect(i=_from_neuron_indexes[nog_indexes], j=_to_neuron_indexes[nog_indexes])
            S_not.u_0 = syn_pars['u0']
            S_not.w = jxx
            network_objects.append(S_not)

        return network_objects

#-----------------------------------------------------------------------------------------------------------------------
# Utility to deal with indexes for Tripartite synapses -- Mirrors the if condition and actions in the TripartiteSynapse
# group -- This is needed mostly in handling recordings of monitors and work on correct indexes
#-----------------------------------------------------------------------------------------------------------------------
def tripartiteIndexing(edges_xx,edges_gxx,edges_xxg,glia_clusters,shared=False,gliot=False):
    # Default values
    from_neuron_indexes = edges_xx[0]
    to_neuron_indexes = edges_xx[1]
    source_syn_indexes = edges_gxx[0]
    to_glia_indexes = edges_gxx[1]
    from_glia_indexes = edges_xxg[0]
    target_syn_indexes = edges_xxg[1]

    # The following modifies the above only if some heterogeneity between tripartite and non-tripartite synapses is detected
    # Indexes of the tripartite synapse
    N_syn = np.shape(edges_xx)[1]
    if (shared and N_syn != np.size(source_syn_indexes)) or (gliot and N_syn != np.size(target_syn_indexes)):
        if not (shared and gliot and np.size(source_syn_indexes)!=np.size(target_syn_indexes)):
            tripartite_indexes = np.union1d(source_syn_indexes,target_syn_indexes).astype(np.int32)
            from_neuron_indexes = edges_xx[0][tripartite_indexes]
            to_neuron_indexes = edges_xx[1][tripartite_indexes]
            # Need also to rearrange the from/to glia indexes according to the permutations by union1d which is rearranging in sorted order
            if gliot:
                from_glia_indexes = edges_xxg[0][np.argsort(target_syn_indexes)]
                target_syn_indexes = np.sort(edges_xxg[1])
            if shared:
                to_glia_indexes = edges_gxx[1][np.argsort(source_syn_indexes)]
                source_syn_indexes = np.sort(edges_gxx[0])
            # -------------------------------------------------------------------------------------------------------
            # Update indexes of synapses according to total tripartite synapses
            # -------------------------------------------------------------------------------------------------------
            # There is a mismatch in fact in terms of syn_indexes as they are passed to the module -- referring to edges[xx] and the internal count in the monitor
            syn_indexes = np.arange(np.size(tripartite_indexes))
            # Turn source/target indexes into real group indexes
            for cn,gc in enumerate(glia_clusters):
                # All glia indexes in the cluster
                glia_in_cluster = np.union1d(from_glia_indexes[np.isin(from_glia_indexes,gc)],to_glia_indexes[np.isin(to_glia_indexes,gc)])
                # All source_synapses in the clustered according to how it is allocated in the group
                to_sidx = np.where(np.isin(from_glia_indexes,glia_in_cluster))[0]
                target_syn_indexes[to_sidx] = syn_indexes[np.isin(tripartite_indexes,target_syn_indexes[to_sidx])]
                from_sidx = np.where(np.isin(to_glia_indexes,glia_in_cluster))[0]
                source_syn_indexes[from_sidx] = syn_indexes[np.isin(tripartite_indexes,source_syn_indexes[from_sidx])]
        else:
            if shared and gliot:
                # This handles the special case where you have some synapses that are only stimulating/modulated by glia
                # TODO: This can be extended to handle the general case of any configuration: currently ONLY handles the case where gliot synapses are a subset of shared synapses
                # TODO: This has been tested only in the special case of 2-cluster network with p_eeg = p_ieg = [0.0,1.0]
                tripartite_indexes,fs_indexes,ts_indexes = np.intersect1d(source_syn_indexes,target_syn_indexes,return_indices=True)
                # Indexes of effective tripartite synapses
                from_neuron_indexes = edges_xx[0][tripartite_indexes]
                to_neuron_indexes = edges_xx[1][tripartite_indexes]
                from_glia_indexes = edges_xxg[0][ts_indexes]
                target_syn_indexes = edges_xxg[1][ts_indexes]
                to_glia_indexes = edges_gxx[0][fs_indexes]
                source_syn_indexes = edges_gxx[1][fs_indexes]
                # -------------------------------------------------------------------------------------------------------
                # Update indexes of synapses according to total tripartite synapses
                # -------------------------------------------------------------------------------------------------------
                # There is a mismatch in fact in terms of syn_indexes as they are passed to the module -- referring to edges[xx] and the internal count in the monitor
                syn_indexes = np.arange(np.size(tripartite_indexes))
                ts_indexes,fs_indexes = [],[]
                # Turn source/target indexes into real group indexes
                for cn,gc in enumerate(glia_clusters):
                    # All glia indexes in the cluster
                    glia_in_cluster = np.union1d(from_glia_indexes[np.isin(from_glia_indexes,gc)],to_glia_indexes[np.isin(to_glia_indexes,gc)])
                    # All target_synapses in the clustered according to how it is allocated in the group
                    to_sidx = np.where(np.isin(from_glia_indexes,glia_in_cluster))[0]
                    # True indexes of targeted synapses in the cluster
                    target_syn_indexes[to_sidx] = syn_indexes[np.isin(tripartite_indexes,target_syn_indexes[to_sidx])]
                    print("WARNING: NOT IMPLEMENTED source_syn_indexes for monitors' handling in the asymmetric case")
                    # from_sidx = np.where(np.isin(to_glia_indexes,glia_in_cluster))[0]
                    # # True indexes of source synapses in the cluster
                    # source_syn_indexes[from_sidx] = syn_indexes[np.isin(tripartite_indexes,source_syn_indexes[from_sidx])]

    return from_neuron_indexes,to_neuron_indexes,from_glia_indexes,to_glia_indexes,target_syn_indexes,source_syn_indexes