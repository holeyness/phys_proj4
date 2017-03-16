# Project 4: Molecular Orbital Theory
# Members: Yueming (Ian) Luo, Erick Orozco, Sydney Holway

import numpy as np
import sympy as smp
import scipy as sp
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter

"""
Preface:
Part A - Huckel theory for pi-molecular orbitals.
For a given matrix H, with Alpha diagnoals and Beta off-diagnoals, we will determine the Eigenvalues and Eigenvectors
for the Huckel Effective Hamiltonian and use them to create an energy level diagram for the electronic configuration
of molecule.
"""

a, b = smp.symbols('a, b')           # Generate symbols so we can use them in our fancy matrix


class Molecule:
    """This class represents a molecule with a Huckel Matrix"""
    def __init__(self, name, huckel, num_pi_electrons):
        self.huckel = huckel
        self.name = name
        self.eigenvalues = []
        self.eigenvectors = []
        self.mega_eigen_array = []
        self.num_pi_electrons = num_pi_electrons
        self.deloc_energy = 0.0
        self.alpha = None
        self.beta = None

    @staticmethod
    def sort_eigs(eig):
        """This is a sorting function that is able to sort the eigenvalues symbolically"""
        return smp.N(eig.subs(a, -1).subs(b, -1))

    def __str__(self):
        """Create the string representation for the molecule"""
        return self.name + ": " + str(self.huckel)

    def set_constants(self, alpha, beta):
        """This function substitutes numeric value in place for Alpha and Beta in the Huckel Matrix"""
        self.huckel = self.huckel.subs(a, alpha).subs(b, beta)

        # Store the resonance integral values
        self.alpha = alpha
        self.beta = beta

    def generate_eigen(self):
        """Finds the eigenvalue and eigenvector for the huckel matrix"""
        # Generates list of tuple (eigenvalues, multiplicity)
        self.eigenvalues = list(self.huckel.eigenvals().keys())
        self.eigenvectors = [x[2] for x in self.huckel.eigenvects()]    # The 3rd element contains the eigenvector
        self.mega_eigen_array = self.huckel.eigenvects()

    def print_eigenvectors(self):
        """Pretty print the list of vectors"""
        for m in self.eigenvectors:
            smp.pprint(m)

    def find_nodes(self):
        """Finds the nodes for the Eigenvector"""

        def find_nodes_helper(eigenvector):
            """ A helper function that count the number of nodes for a specific eigenvector"""

            nodes = 0                       # Number of Nodes
            for i in range(len(eigenvector) - 1):
                if eigenvector[i] * eigenvector[i + 1] < 0:
                    nodes += 1              # We have a node
            return nodes

        result = []
        for eig_set in self.mega_eigen_array:
            # Eig_Set = [eigenvalue, multiplyer, [eigenvector]]
            result.append((eig_set[0], find_nodes_helper(eig_set[2][0])))

        # Result = [(ev, num_nodes)...]
        return result

    def energy_level_plot(self):
        """Plots the energy levels and denoate spin for the electrons"""
        assert(self.alpha is not None and self.beta is not None)        # Make sure alpha and beta are defined

        self.generate_eigen()

        # Get the unicode arrow
        down_arrow = u'$\u2193$'
        up_arrow = u'$\u2191$'

        electrons_used = self.num_pi_electrons
        for eig in sorted(self.eigenvalues, key=self.sort_eigs):
            plt.axhline(eig)                                 # Draw the eigenvalues as lines on the graph
            # Fill up to two electrons per level, from bottom up
            if electrons_used > 1:
                plt.plot(-0.8, eig, linestyle='none', marker=up_arrow, markersize=15)
                plt.plot(-0.2, eig, linestyle='none', marker=down_arrow, markersize=15)
                electrons_used -= 2
            elif electrons_used == 1:
                plt.plot(-0.8, eig, linestyle='none', marker=up_arrow, markersize=15)
                electrons_used -= 1

            else:
                pass

        plt.title('Energy Level Plot for ' + str(self.name))
        plt.xlim(-1, 0)                                         # Format the Graph
        plt.xticks([])                                          # Hide the x-axes
        plt.ylabel('Energy')
        plt.show()

    def find_deloc_energy(self):

        if self.beta == 0.0:
            # Find symbolic solution
            deloc_energy = 0.47 * b

        else:
            deloc_energy = 0.47 * self.beta

        num_electrons = self.num_pi_electrons
        eig_index = 0

        while num_electrons > 0:
            if num_electrons > 1:
                deloc_energy += 2 * sorted(self.eigenvalues, key=self.sort_eigs)[eig_index]
                eig_index += 1
                num_electrons -= 2
            elif num_electrons == 1:
                deloc_energy += sorted(self.eigenvalues, key=self.sort_eigs)[eig_index]
                eig_index += 1
                num_electrons -= 1
            else:
                pass

        self.deloc_energy = deloc_energy

        return self.deloc_energy



"""Part A"""

butadiene_huckel = smp.Matrix([[a, b, 0, 0], [b, a, b, 0],
                               [0, b, a, b], [0, 0, b, a]])       # Create our matrix for the molecule
butadiene = Molecule("Butadine", butadiene_huckel, 4)      # Create the matrix
butadiene.generate_eigen()                              # Generate the eigenvalues and eigenvector
butadiene.energy_level_plot()
print(butadiene.find_deloc_energy())