"""
Original Authors: David Corbo & Christian Frech 
File Creation Date: May 11, 2020
Development Group: Genetically Engineered Materials Science & Engineering Center
Description: Contains methods and routines for creating adjacencies for peptide-water network and 
    calculating standard deviation of energy states
"""

import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from numpy import linalg as LA
import math
from matplotlib import cm
import matplotlib.colors as mcolors
import imageio
import image
import cupy as cp
from mpl_toolkits.mplot3d import Axes3D


class PDBAtom(object):
    """ Class to represent a single atom's position and state at a frame
    
    Attributes:
        _valence_dict (dict{str: int}): A dictionary of valence electron count per element
        x (float): The x coordinate of the atom
        y (float): The y coordinate of the atom
        z (float): The z coordinate of the atom
        valence_count (int): Number of valence electrons in the atom
    """
    
    
    _valence_dict = {'C': 4,
                     'H': 1,
                     'N': 5,
                     'O': 6,
                     'S': 6}
    
    _electroneg_dict = {'C': 2.55,
                        'H': 2.2,
                        'N': 3.04,
                        'O': 3.44,
                        'S': 2.58}
    
    def __init__(self, string):
        """ Standard PDB file format
        ATOM    277  O1  LYS A  14      21.138  -0.865  -4.761  1.00  0.00           O1-
        """
#       Coordinate Parser 
        self.x = float(string[30:38].strip())
        self.y = float(string[38:46].strip())
        self.z = float(string[46:54].strip())
        
#       Element and Valence Electron Number Parser
        self.element_spec = string[77:].strip()
        mod = 0
        if self.element_spec.endswith(('-', '+')):
            self.element_sym = self.element_spec[:-2].strip()
            mod = int(self.element_spec[-2])
            mod *= (-1, 1)[self.element_spec.endswith('-')]
        else:
            self.element_sym = self.element_spec.strip()
        self.valence_count = PDBAtom._valence_dict.get(self.element_sym)
        if self.valence_count is None:
            raise TypeError('Used an element that is not in the valence dictionary')
        else:
            self.valence_count += mod
        self.electronegativity = PDBAtom._electroneg_dict.get(self.element_sym)

#       Structure Index Parser
        self.structure_index = int(string[24:26].strip())

#       Atom Letter Parser (indicates which sidechains are interacting with which parts of the backbone)
        self.atom_letter = string[14:15].strip()

class Adj_Mats(object):
    """ Class to represent a series of adjacency matrices
    
    Attributes:
        file (str): The path of the pdb file to be parsed
        valence_list (Array[Array[int]]): Stores the number of valence electrons in all atoms in every frame
        distance_graphs (Array[Array[Array[int]]]): The series of distance matrices of the atoms in the evolution
        adjacenecy_graphs (Array[Array[Array[int]]]): The series of adjacency matrices of the atoms in the evolution
        elec_adjacency_graphs (Array[Array[Array[int]]]): The series of adjacency matrices of electrons in the evolution
        
    Methods:
        set_atom_dists: Used to set the distance_graphs attribute
        set_atom_adj: Used to set the adjacency_graphs attribute
        get_atom_dists: Used to parse the pdb file to create a distance_graphs object
        get_atom_adj: Used to set an adjacency threshold on the distance matrices and make adjacency matrices
        get_elec_adj: Used to convert atom adjacency matrix to corresponding electron adjacency matrix given 
                    number of valence electrons in each atom 
        entropyCalculation: Used to calculate continuous differential entropy of electron adjacency matrix
    """

    def __init__(self, pdb, batchframes):
        self.file = pdb
        self.valence_list = np.zeros(1, int)
        self.distance_graphs = np.zeros(1, int)
        self.adjacency_graphs = np.zeros(1, int)
        self.elec_adjacency_graphs = np.zeros(1, int)
        self.elneg_adj = np.zeros(1, int)
        self.eigenvalues = None
        self.bin_probs = None
        self.entropy = None
        self.energy = None
        self.cont_ent = None
        self.batchframes = batchframes 

    def set_atom_dists(self, new_dists):
        self.distance_graphs = new_dists
        
    def set_atom_adj(self, new_adj):
        self.adjacency_graphs = new_adj
    
    def get_atom_dists(self):
        if os.path.isfile(self.file):
            pdb_file = open(self.file,'r')
        else:
            raise OSError('File {} does not exist'.format(self.file))

        lineno = 0
        frames = []
        atoms = []
        atom_types = []
        val_frames = []
        val_atoms = []
        atoms_list = []
        structure_list = []
        atom_letter_list = []
        atom_types = []
        structure_number = []
        atom_letters = []


        numberof_coordinates = 218 
        index = 0

        lines = [line for line in pdb_file]
        while index <(len(lines)):
            line = lines[index]
            if line.startswith('MODEL'):
                pdbframe = int(line[9:].strip())
                if pdbframe not in self.batchframes:
                    index += numberof_coordinates
                    atoms = []          
                    val_atoms = []
                    atom_types = []
                    structure_number = []
                else: 
                    pass

            elif line.startswith('ATOM'):
                at_obj = PDBAtom(line)
                atoms.append([at_obj.x, at_obj.y, at_obj.z])
                val_atoms.append(at_obj.valence_count)
                atom_types.append(at_obj.element_spec)
                structure_number.append(at_obj.structure_index)
                atom_letters.append(at_obj.atom_letter)

            elif line.startswith('END'):
                frames.append(atoms)
                atoms = []
                val_frames.append(val_atoms)
                val_atoms = []
                atoms_list.append(atom_types)
                atom_types = []
                structure_list.append(structure_number)
                structure_number = []
                atom_letter_list.append(atom_letters)
                atom_letters = []

            index += 1
        pdb_file.close()
        print("len frames: " +str(len(frames)))
        base = np.zeros((len(frames), len(frames[0]), 3))
        for i in range(len(frames)):
            for j in range(len(frames[i])):
                for k in range(len(frames[i][j])):
                    base[i][j][k] = frames[i][j][k]
        dists = np.reshape(base, (len(frames), 1, len(frames[0]), 3)) - np.reshape(base, (len(frames), len(frames[0]), 1, 3))  
        dists = dists**2
        dists = dists.sum(3)
        dists = np.sqrt(dists)

        self.structure_list = structure_list
        self.atoms_list = atoms_list
        self.atom_letters = atom_letter_list
        self.valence_list = val_frames
        self.distance_graphs = dists

        return self.distance_graphs
    
    def get_atom_adj(self,s,t):
        self.get_atom_dists()

        self.adjacency_graphs = ((self.distance_graphs < t) & (self.distance_graphs > s)).astype(int)

        #Creating same-structure interactions and screening through secondary bonds:
        for frame in range(len(self.adjacency_graphs)):
            for i in range(len(self.adjacency_graphs[frame])):
                for j in range(len(self.adjacency_graphs[frame][i])):
                    '''
                    PEPTIDE ADJACENCY
                    '''
                    atom_a = self.atoms_list[frame][i]
                    atom_b = self.atoms_list[frame][j]
                    struc_index_a = self.structure_list[frame][i]
                    struc_index_b = self.structure_list[frame][j]
                    atom_letter_a = self.atom_letters[frame][i]
                    atom_letter_b = self.atom_letters[frame][j]
                    interaction = self.adjacency_graphs[frame][i][j]

                    if (struc_index_a == struc_index_b):
                        # Creating bonds between same-lettered (carbon & hydrogen) atoms
                        if (atom_letter_a == atom_letter_b):
                            self.adjacency_graphs[frame][i][j] = 1
                        elif ((atom_letter_a.isdigit() == True) & (atom_b == 'N')):
                            self.adjacency_graphs[frame][i][j] = 1
                        elif  ((atom_letter_b.isdigit() == True) & (atom_a == 'N')):
                            self.adjacency_graphs[frame][i][j] = 1
                        # Creating bonds between nitrogen and 'N' hydrogen atoms
                        elif ((atom_letter_a == 'N') & (atom_b == 'N')) or ((atom_letter_b == 'N') & (atom_a == 'N')):
                            self.adjacency_graphs[frame][i][j] = 1
                        # Creating bonds between nitrogen and carbon atoms
                        elif ((atom_a == 'N') & (atom_b == 'C') & (atom_letter_b == 'A')):
                            self.adjacency_graphs[frame][i][j] = 1
                        elif ((atom_b == 'N') & (atom_a == 'C') & (atom_letter_a == 'A')):
                            self.adjacency_graphs[frame][i][j] = 1
                        # Creating bonds between oxygens and carbon atoms for intermediate alanines
                        elif ((atom_a == 'O') & (atom_b == 'C') & (not atom_letter_b)):
                            self.adjacency_graphs[frame][i][j] = 1
                        elif ((atom_b == 'O') & (atom_a == 'C') & (not atom_letter_a)):
                            self.adjacency_graphs[frame][i][j] = 1
                        # Creating bonds between oxygen and carbon atoms for last alanine
                        elif (((atom_letter_a == 'T') & (not atom_letter_b)) or ((atom_letter_b == 'T') & (not atom_letter_a))) & ((atom_a == 'C') or (atom_b =='C')):
                            self.adjacency_graphs[frame][i][j] = 1
                        # Creating bonds between first a & b carbon atoms
                        elif (atom_a == atom_b == 'C') & (((atom_letter_a == 'A') & (atom_letter_b == 'B')) or ((atom_letter_a == 'B') & (atom_letter_b == 'A'))):
                            self.adjacency_graphs[frame][i][j] = 1
                        # Creating bonds between a & no-letter carbon atoms
                        elif (atom_a == atom_b == 'C') & (((atom_letter_a == 'A') & (not atom_letter_b)) or ((not atom_letter_a) & (atom_letter_b == 'A'))):
                            self.adjacency_graphs[frame][i][j] = 1
                        # No interation
                        else:
                            self.adjacency_graphs[frame][i][j] = 0 

                    else:
                        # Creating hydrogen bonds between nitrogen and hydrogen
                        if ((((atom_a=='N') & (atom_b=='H')) or ((atom_a=='H') & (atom_b=='N'))) & (interaction==1)):
                            self.adjacency_graphs[frame][i][j] = 1
                        # Creating hydrogen bonds between oxygen and hydrogen
                        elif ((((atom_a=='O') & (atom_b=='H')) or ((atom_a=='H') & (atom_b=='O'))) & (interaction==1)):
                            self.adjacency_graphs[frame][i][j] = 1
                        # Creating bonds between carbons and nitrogens between alanines 
                        elif ((struc_index_a == (struc_index_b + 1)) & (atom_b == 'C') & (atom_a == 'N') & (not atom_letter_b)):
                            self.adjacency_graphs[frame][i][j] = 1
                        elif ((struc_index_b == (struc_index_a + 1)) & (atom_a == 'C') & (atom_b == 'N') & (not atom_letter_a)):
                            self.adjacency_graphs[frame][i][j] = 1
                        # No interation
                        else:
                            self.adjacency_graphs[frame][i][j] = 0

        ### PEPTIDE TEST CASES ###
        print('testing')

        ###N-CA###
        assert(self.adjacency_graphs[0][22][24] == 1)
        assert(self.adjacency_graphs[0][24][22] == 1)
        ###HB-CB###
        assert(self.adjacency_graphs[0][38][36] == 1)
        assert(self.adjacency_graphs[0][36][38] == 1)
        ##CA-CB###
        assert(self.adjacency_graphs[0][36][34] == 1)
        assert(self.adjacency_graphs[0][34][36] == 1)
        ###C-O###
        assert(self.adjacency_graphs[0][31][30] == 1)
        assert(self.adjacency_graphs[0][30][31] == 1)
        ###CA-C###
        assert(self.adjacency_graphs[0][30][24] == 1)
        assert(self.adjacency_graphs[0][24][30] == 1)
        ###N-HN###
        assert(self.adjacency_graphs[0][13][12] == 1)
        assert(self.adjacency_graphs[0][12][13] == 1)
        ###N-H1,2,3###
        assert(self.adjacency_graphs[0][0][2] == 1)
        assert(self.adjacency_graphs[0][2][0] == 1)
        ###NOT C-CB###
        assert(self.adjacency_graphs[0][16][20] == 0)
        assert(self.adjacency_graphs[0][20][16] == 0)
        ###Inter-Alanine C-N###
        assert(self.adjacency_graphs[0][22][20] == 1)
        assert(self.adjacency_graphs[0][20][22] == 1)

        return self.adjacency_graphs
    
    def get_elec_adj(self):
        #if len(self.adjacency_graphs) == 1:
        self.get_atom_adj(lowerlimit,upperlimit)
            
        total_val = 0
        
        for i in range(len(self.valence_list[0])):
            total_val += self.valence_list[0][i]
        valencelistframes = len(self.valence_list)
        self.elec_adjacency_graphs = np.zeros((len(self.adjacency_graphs), total_val, total_val))
        curr_n, curr_m = 0, 0
        
        for i in range(len(self.adjacency_graphs)):
            print(i)
            for j in range(len(self.adjacency_graphs[0])):
                for b in range(self.valence_list[i][j]):
                    for k in range(len(self.adjacency_graphs[0][0])):
                        for a in range(self.valence_list[i][k]):
                            self.elec_adjacency_graphs[i][curr_n][curr_m] = self.adjacency_graphs[i][j][k]
                            curr_m += 1
                    curr_m = 0
                    curr_n += 1
            curr_n = 0
        
        return self.elec_adjacency_graphs
    
    def make_eigenvalues(self, hamiltonian_iter=10):
        self.elec_adjacency_graphs=cp.array(self.get_elec_adj())
        elec_count = len(self.elec_adjacency_graphs[0])
        self.eigenvalues = cp.zeros((len(self.elec_adjacency_graphs), hamiltonian_iter, elec_count))
        cp.cuda.Stream.null.synchronize()
        for frame in range(len(self.elec_adjacency_graphs)):
            for i in range(hamiltonian_iter):
                r = cp.random.normal(size=(elec_count, elec_count))
                cp.cuda.Stream.null.synchronize()
                rt = cp.transpose(r)
                cp.cuda.Stream.null.synchronize()
                h = (r + rt) / cp.sqrt(2 * elec_count)   
                cp.cuda.Stream.null.synchronize()       
                adj_r = self.elec_adjacency_graphs[frame] * h
                vects_values = cp.linalg.eigh(adj_r)
                eigs = cp.ndarray.tolist(vects_values[0])
                cp.cuda.Stream.null.synchronize()
                eigs.sort()
                for value in range(len(eigs)):
                    self.eigenvalues[frame][i][value] = eigs[value]
                cp.cuda.Stream.null.synchronize()
        return self.eigenvalues

    def get_spacings(self,types):
        eigenvalues = self.make_eigenvalues()
        allframes_spacing=cp.zeros(len(eigenvalues), len(eigenvalues[0]))
        eigenvalues=cp.asnumpy(eigenvalues)
        
        if types=='all':
            medians=[]
            nexttomedian=[]
            spacings=[]
            allspacings=cp.zeros((len(eigenvalues[0]),len(eigenvalues[0][0])-1))
            cp.cuda.Stream.null.synchronize()
            for i in range(len(eigenvalues[0])):
                for j in range(len(eigenvalues[0][0])-1):
                    allspacings[i][j]=(eigenvalues[0][i][j+1]-eigenvalues[0][i][j])
            allspacings=allspacings.transpose()
            counts, binedges = cp.histogram(allspacings)
            cp.cuda.Stream.null.synchronize()
            return [counts, allspacings]

        else:
            for frame in range(len(self.elec_adjacency_graphs)):
                medians=cp.zeros(len(eigenvalues[frame]))
                nexttomedian=cp.zeros(len(eigenvalues[frame]))
                spacings=cp.zeros(len(eigenvalues[frame]))
                for i in range(len(eigenvalues[frame])):
                    #Calculating medians and converting eigenvalue array to 1xn list:
                    medians[i] = np.median(eigenvalues[frame][i])

                    if len(eigenvalues[frame][i])%2==0:
                        nexttomedian[i] = (eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+1]+eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+2])/2
                    else:
                        nexttomedian[i] = eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+1]
                #Calculating median and median+1 spacings, along with spacing standard deviation:
                for i in range(len(medians)):
                    spacings[i] = abs(nexttomedian[i]-medians[i])
                #Entering spacings into array of all spacings
                for space in range(len(spacings)):
                    allframes_spacing[frame][space] = spacings[space]
            return allframes_spacing

    def get_spacings_and_stdevs(self,types):
        eigenvalues = self.make_eigenvalues()
        stdevs=[]

        if types=='all':
            for frame in range(len(eigenvalues)):
                oneframe_stdevs = []
                allspacings = cp.zeros((len(eigenvalues[frame]), len(eigenvalues[frame][0]- 1) ))
                cp.cuda.Stream.null.synchronize()
                for i in range(len(eigenvalues[frame])):
                    for j in range(len(eigenvalues[frame][i]) - 1):
                        allspacings[i][j] = (eigenvalues[frame][i][j+1] - eigenvalues[frame][i][j])
                allspacings=allspacings.transpose()
                for i in range(len(allspacings)):
                    oneframe_stdevs.append(np.std(cp.asnumpy(allspacings[i])))
                stdevs.append(oneframe_stdevs)
            return [stdevs, allspacings]

        else:
            print("len eigs: " + str(len(eigenvalues)))
            for frame in range(len(eigenvalues)):
                eigenvalues=cp.asnumpy(eigenvalues)
                cp.cuda.Stream.null.synchronize()
                spacings=[]
                medians=[]
                nexttomedian=[]
                for i in range(len(eigenvalues[frame])):
                #Calculating medians and converting eigenvalue array to 1xn list:
                    medians.append(np.median(eigenvalues[frame][i]))
                    if len(eigenvalues[frame][i])%2==0:
                        nexttomedian.append((eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+1]+eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+2])/2)
                    else:
                        nexttomedian.append(eigenvalues[frame][i][math.floor(len(eigenvalues[frame])/2)+1])
                #Calculating median and median+1 spacings, along with spacing standard deviation:
                for i in range(len(medians)):
                    spacings.append(nexttomedian[i]-medians[i])
                stdevs.append(np.std(spacings))
            return stdevs


'''
Initialize the frames and distances to iterate through:
lowerlimit/upperlimit: lower and upper limit of bonding distances
frameindices: which frames are used for stdev calculation
types: which type of eigenvalue spacings to use 
    ('all_frames'=3D surface plot of bond distances (X) vs. frame number (Y) vs. standard deviation of spacings (Z))
    ('all'=3D surface plot of bond distances (X) vs. eigenvalue pair number (Y) vs. standard deviation of spacings (Z))
    ('median'=2D line plot of bond distances (X) vs. standard deviation of spacings (Y))
'''
lowerlimit=4
upperlimit_list=np.arange(lowerlimit,10,3)
allframes_stdevs=[]
frame_period = 5000
batchframes = np.arange(1,20002,frame_period)
print("batchframes: "+ str(len(batchframes)))
batchframesextend = np.arange(1,10002,frame_period)
print("extend: " + str(len(batchframesextend)))
framesindices = np.concatenate((batchframes, (batchframesextend + 20000)))

types='all'
frames = []
filenames = []


'''
Calling methods from Adj_Mats class, iterating through frames (frameindex) and through bonding distances (upperlimit) to
append the standard deviations to full 2D list (allframes_stdevs)

Specify PDB files to be used under filname1 and filename2
'''

oneframe_stdevs=[]
hydrogenbonds_array=[]
lowerlimit=1

filename1= "A21_peponly_100ns.pdb"
filename2 = "peptide.pdb"

for upperlimit in upperlimit_list:
    if __name__ == "__main__":
        print("upperlimit: " +str(upperlimit))
        if (types=='all'):
            one_dist_stdevs = []
            file = open(filename1)
            batch = Adj_Mats(filename1, batchframes)
            batch_stdev = batch.get_spacings_and_stdevs(types)
            one_dist_stdevs.extend(batch_stdev[0])
            file = open(filename2)
            batch2 = Adj_Mats(filename2, batchframesextend)
            batch_stdev = batch2.get_spacings_and_stdevs(types)
            one_dist_stdevs.extend(batch_stdev[0])
            allspacings=batch_stdev[1]

        else: 
            one_dist_stdevs = []
            file = open(filename1)
            batch = Adj_Mats(filename1, batchframes)
            batch_stdev = batch.get_spacings_and_stdevs(types)
            one_dist_stdevs.extend(batch_stdev)
            print("len one_dist_stdevs: " + str(len(one_dist_stdevs)))
            file = open(filename2)
            batch2 = Adj_Mats(filename2, batchframesextend)
            batch_stdev = batch2.get_spacings_and_stdevs(types)
            one_dist_stdevs.extend(batch_stdev)
            print("len one_dist_stdevs: " + str(len(one_dist_stdevs)))

    allframes_stdevs.append(one_dist_stdevs)

allframes_stdevs = cp.asnumpy(allframes_stdevs)
allframes_stdevs = np.transpose(allframes_stdevs, (1, 0, 2))


'''
MAKING PLOTS (different plots for different input of types variable)
'''

if (types=='all_frames'):
    Z = allframes_stdevs
    fig = plt.figure(figsize=plt.figaspect(0.5))

    #Creating X-axis for 3D plots (frames)
    xx = framesindices
    iterations = upperlimit_list
    yaxis_label = 'Cut-Off Distance For Interactions'
    yy = upperlimit_list
    X, Y = np.meshgrid(xx,yy)
    print(X.shape)
    print(Y.shape)
    print(Z.shape)
    ax = fig.add_subplot(1, 2, 1, projection='3d')
    surf = ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
    fig.colorbar(surf, shrink=0.5, aspect=10)
    ax.set_xlabel('Frame index')
    ax.set_ylabel(yaxis_label)
    ax.set_zlabel('Spacings Standard Deviation')
    plotname = 'pep_test_stdev_allframes' + '.png'
    plt.savefig(plotname, dpi=95)

elif (types=='all'):

    #XY-plane
    yy = upperlimit_list
    xx=np.arange(0, (len(allspacings)), 1)
    X,Y = np.meshgrid(xx,yy)

    for i in range(len(allframes_stdevs)):
        #Z value Distributions
        Z = allframes_stdevs[i]
        print(X.shape)
        print(Y.shape)
        print(Z.shape)

        #Make 3D surface plot
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X, Y, Z, s = 6)

        ax.set_xlabel('Eigenvalue Spacing Pair #')
        ax.set_ylabel('Interaction Distance Cutoff (Angstroms)')
        ax.set_zlabel('Standard Deviation of Spacings')

        plt.savefig('pep_test_stdev_all.png', dpi=95)
        filename='pep_test_stdev_all_'+str(framesindices[i])+'.png'
        filenames.append(filename)
        plt.savefig(filename, dpi=95)
        plt.clf()
    
    images=[]
    for name in filenames:
        images.append(imageio.imread(name))
    imageio.mimsave('pep_test_stdev_all.gif', images, duration=1)
