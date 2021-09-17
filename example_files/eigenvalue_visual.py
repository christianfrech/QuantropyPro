"""
Original Authors: David Corbo & Christian Frech 
File Creation Date: May 11, 2020
Development Group: Genetically Engineered Materials Science & Engineering Center
Description: Contains methods and routines for creating adjacencies for peptide-water network and 
    generating associated eigenvalue distribution visualizations
"""


import cupy as cp
import numpy as np
import os
import sys
import seaborn as sns
import matplotlib.pyplot as plt
from numpy import linalg as LA
import xlsxwriter


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
        make_eigenvalues: Used to calculate eigenvalues of electron adjacency matrix
                
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
                    print(pdbframe)
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
        print(len(frames))
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
        self.get_atom_adj(lowerlimit,upperlimit)
        
        print('getting elec adj')

        total_val = 0
        
        for i in range(len(self.valence_list[0])):
            total_val += self.valence_list[0][i]
        valencelistframes = len(self.valence_list)
        
        self.elec_adjacency_graphs = np.zeros((len(self.adjacency_graphs), total_val, total_val))
        
        curr_n, curr_m = 0, 0
        
        for i in range(len(self.adjacency_graphs)):
            for j in range(len(self.adjacency_graphs[0])):
                for b in range(self.valence_list[i][j]):
                    for k in range(len(self.adjacency_graphs[0][0])):
                        for a in range(self.valence_list[i][k]):
                            self.elec_adjacency_graphs[i][curr_n][curr_m] = self.adjacency_graphs[i][j][k]
                            curr_m += 1
                    curr_m = 0
                    curr_n += 1
            curr_n = 0  
        
        print('done getting elec adj')
        return self.elec_adjacency_graphs
    
    def make_eigenvalues(self, hamiltonian_iter, types):
        self.elec_adjacency_graphs=cp.array(self.get_elec_adj())
        elec_count = len(self.elec_adjacency_graphs[0])
        self.eigenvalues = cp.zeros((len(self.elec_adjacency_graphs), hamiltonian_iter, elec_count))
        cp.cuda.Stream.null.synchronize()
        for frame in range(len(self.elec_adjacency_graphs)):
            print(frame)
            for i in range(hamiltonian_iter): 
                r = cp.random.normal(size=(elec_count, elec_count))
                cp.cuda.Stream.null.synchronize()
                rt = cp.transpose(r)
                cp.cuda.Stream.null.synchronize()
                h = (r + rt) / cp.sqrt(2 * elec_count)   
                cp.cuda.Stream.null.synchronize()       
                adj_r = self.elec_adjacency_graphs[frame] * h
                vects_values = cp.linalg.eigh(adj_r)
                eigs = vects_values[0]
                cp.cuda.Stream.null.synchronize()
                eigs.sort()
                self.eigenvalues[frame][i] = eigs

        return self.eigenvalues

def binSizes(values,binscount):
    binvalues=[]
    width=abs(max(values)-min(values))
    for i in range(binscount):
        binvalues.append(sum([min(values),i*(width/binscount)]))
    return binvalues

"""
Set PDB files, names, and frames to iterate through (batchframes), number of bins,
number of interactions, names of frames (for histogram), and interaction cutoff distance
"""
filename1 = "A21_peponly_100ns.pdb"
filename2 = "peptide.pdb"
names = ['0ns', '100ns', '200ns']
types = 'all_frames'
binscount = 1000
num_of_iterations = 10
lowerlimit = 0
upperlimit = 5
batchframes = [1,15000]
batchframesextend = [9999]

eigenvalues = []
flattened_eigen = []


'''
Calling methods from Adj_Mats class for a given set of frames (batchframes) and bonding distances (upperlimit) to
append the resultant eigenvalues to 'eigenvalues' and flatten each frame into one list in the 2D array 'flattened_eigen'
'''

if __name__ == "__main__":
    firsthalf_spacings = []
    file = open(filename1)
    batch = Adj_Mats(filename1, batchframes)
    batch_eigs = np.ndarray.tolist(cp.asnumpy(batch.make_eigenvalues(num_of_iterations,types)))
    file = open(filename2)
    batch2 = Adj_Mats(filename2, batchframesextend)
    batch2_eigs = np.ndarray.tolist(cp.asnumpy(batch2.make_eigenvalues(num_of_iterations,types)))
    batch_eigs.extend(batch2_eigs)
    eigenvalues = np.array(batch_eigs)
    for frame in range(len(eigenvalues)):
        flattened_eigen.append(np.array(eigenvalues[frame]).flatten())
        
    subeigenvalues=flattened_eigen
    fig, axes = plt.subplots(1,1)

"""
Plotting 2D probability density graphs for eigenvalues 
"""

for i in range(len(flattened_eigen)):
    axis=axes
    eigenvaluebinvalues=binSizes(subeigenvalues[i], binscount)
    weighting = np.ones_like(subeigenvalues[i]) / len(subeigenvalues[i])
    axis.hist(subeigenvalues[i], bins = eigenvaluebinvalues, alpha=0.5, weights=weighting)

axis.set_ylabel('Probability')
axis.set_xlabel('Eigenvalue')

plt.xlabel('Eigenvalue')
plt.ylabel('Probability')
plt.legend(names,loc=2)
plt.savefig('eigenvalue_test.png', dpi=95)