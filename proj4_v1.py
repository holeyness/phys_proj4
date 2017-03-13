# Project 4: Molecular Orbital Theory
# Members: Yueming (Ian) Luo, Erick Orozco, Sydney Holway

import numpy as np
import sympy as smp
import scipy as sp
import math

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
    def __init__(self, name, huckel):
        self.huckel = huckel
        self.name = name
        self.eigenvalues = []
        self.eigenvectors = []
        self.mega_eigen_array = []

    def __str__(self):
        """Create the string representation for the molecule"""
        return self.name + ": " + str(self.huckel)

    def set_constants(self, alpha, beta):
        """This function substitutes numeric value in place for Alpha and Beta in the Huckel Matrix"""
        self.huckel = self.huckel.subs(a, alpha)
        self.huckel = self.huckel.subs(b, beta)

    def generate_eigen(self):
        """Finds the eigenvalue and eigenvector for the huckel matrix"""
        self.eigenvalues = self.huckel.eigenvals()                      # Generates list of eigenvalues: multiplicity
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


"""Part A"""

butadiene_huckel = smp.Matrix([[a, b, 0, 0], [b, a, b, 0],
                               [0, b, a, b], [0, 0, b, a]])       # Create our matrix for the molecules
butadiene = Molecule("butadine", butadiene_huckel)      # Create the matrix
butadiene.generate_eigen()                              # Generate the eigenvalues and eigenvector
butadiene.set_constants(-5, -10)
butadiene.generate_eigen()                              # Generate the eigenvalues and eigenvector
print(butadiene.find_nodes())
