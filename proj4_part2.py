"""PART 2"""

# from proj4_v1 import Molecule
import sympy as smp
import numpy as np
import sympy as smp
import scipy as sp
import math
import matplotlib.pyplot as plt
from collections import OrderedDict
from operator import itemgetter

a, b = smp.symbols('a, b')  # Generate symbols so we can use them in our fancy matrix


class Molecule:
    """This class represents a molecule with a Huckel Matrix"""

    def __init__(self, name, huckel, num_pi_electrons, num_carbons, num_double_bonds):
        self.huckel = huckel
        self.name = name
        self.num_carbons = num_carbons
        self.eigenvalues = []
        self.eigenvectors = []
        self.normalized_eigenvectors = []
        self.mega_eigen_array = []
        self.num_pi_electrons = num_pi_electrons
        self.deloc_energy = 0.0
        self.alpha = None
        self.beta = None
        self.charge_density = []
        self.bond_order = []
        self.num_additional_connections = 0
        self.connections = []
        self.num_double_bonds = num_double_bonds

        # This is our internal data structure that contains [(eig_value, # of Electrons)]
        self.e_per_energy_lvl = []

    @staticmethod
    def sort_eigs(eig):
        """This is a sorting function that is able to sort the eigenvalues symbolically"""
        return smp.N(eig[0].subs(a, -1).subs(b, -1))

    def __str__(self):
        # TODO: Reeformat with the new cool method
        """Create the string representation for the molecule"""
        return self.name + ": " + str(self.huckel)

    def set_constants(self, alpha, beta):
        """This function substitutes numeric value in place for Alpha and Beta in the Huckel Matrix"""
        self.huckel = self.huckel.subs(a, alpha).subs(b, beta)

        # Store the resonance integral values
        self.alpha = alpha
        self.beta = beta

    def generate_huckel(self):
        """generates huckel matrix for linear carbon chain"""
        N = self.num_carbons
        huckel = smp.zeros(N)
        for i in range(N):
            huckel[i, i] = a
            if i == 0:
                huckel[i, i + 1] = b
            elif i == N - 1:
                huckel[i, i - 1] = b
            else:
                huckel[i, i + 1] = b
                huckel[i, i - 1] = b
        self.huckel = huckel

    def add_connections(self, connections):
        """adds connections to the Huckel matrix, takes a list of tuples and inserts beta into huckel at the specified coordinates"""
        self.connections = connections
        for i in range(len(connections)):
            self.huckel[connections[i][0] - 1, connections[i][1] - 1] = b
            self.huckel[connections[i][1] - 1, connections[i][0] - 1] = b
        self.num_additional_connections += len(connections)

    def delete_connections(self, connections):
        """deletes connections to the Huckel matrix, takes a list of tuples and inserts zero into huckel at the specified coordinates"""
        for i in range(len(connections)):
            self.huckel[connections[i][0] - 1, connections[i][1] - 1] = 0
            self.huckel[connections[i][1] - 1, connections[i][0] - 1] = 0
        self.num_additional_connections -= len(connections)

    def generate_eigen(self):
        """Finds the eigenvalue and eigenvector for the huckel matrix"""
        self.e_per_energy_lvl = []
        self.eigenvectors = []
        self.eigenvalues = []

        # Generates list of tuple (eigenvalues, multiplicity)
        self.mega_eigen_array = sorted(self.huckel.eigenvects(), key=self.sort_eigs)
        for eig in self.mega_eigen_array:
            self.eigenvalues.append(eig[0])
            self.eigenvectors.append(eig[2])

        # Compile our electron per energy level array
        electrons_available = self.num_pi_electrons
        for eig in self.mega_eigen_array:
            eig_val = eig[0]
            multiplicity = eig[1]

            if electrons_available >= (multiplicity * 2):
                self.e_per_energy_lvl.append(tuple([eig_val, 2 * multiplicity]))
                electrons_available -= 2 * multiplicity
            else:
                self.e_per_energy_lvl.append(tuple([eig_val, electrons_available]))
                electrons_available -= electrons_available

    def print_eigenvectors(self):
        """Pretty print the list of vectors"""
        for m in self.eigenvectors:
            smp.pprint(m)

    def find_nodes(self):
        """Finds the nodes for the Eigenvector"""

        def find_nodes_helper(eigenvector):
            """ A helper function that count the number of nodes for a specific eigenvector"""

            nodes = 0  # Number of Nodes
            for i in range(len(eigenvector) - 1):
                if eigenvector[i] * eigenvector[i + 1] < 0:
                    nodes += 1  # We have a node
            return nodes

        result = []
        for eig_set in self.mega_eigen_array:
            # Eig_Set = [eigenvalue, multiplyer, [eigenvector]]
            result.append((eig_set[0], find_nodes_helper(eig_set[2][0])))

        # Result = [(ev, num_nodes)...]
        return result

    def energy_level_plot(self):
        """Plots the energy levels and denoate spin for the electrons"""
        assert (self.alpha is not None and self.beta is not None)  # Make sure alpha and beta are defined

        # Get the unicode arrow
        down_arrow = u'$\u2193$'
        up_arrow = u'$\u2191$'

        electrons_used = self.num_pi_electrons
        max_multiplicity = max(self.mega_eigen_array, key=itemgetter(1))[1]
        for eig in sorted(self.mega_eigen_array, key=self.sort_eigs):
            eig_val = eig[0].subs(a, self.alpha).subs(b, self.beta)
            if eig[1] == 1:
                plt.axhline(eig[0].subs(a, self.alpha).subs(b, self.beta))  # Draw the eigenvalues as lines on the graph
            else:
                for i in range(eig[1]):
                    plt.plot([i, i + 0.95], 2 * [eig_val],
                             color=np.random.rand(3, ))

            # Fill up to two electrons per level per line, from bottom up
            if eig[1] == 1:
                if electrons_used > 1:
                    plt.plot(0.2 * max_multiplicity, eig_val, linestyle='none', marker=up_arrow, markersize=15)
                    plt.plot(0.8 * max_multiplicity, eig_val, linestyle='none', marker=down_arrow, markersize=15)
                    electrons_used -= 2
                elif electrons_used == 1:
                    plt.plot(0.2 * max_multiplicity, eig_val, linestyle='none', marker=up_arrow, markersize=15)
                    electrons_used -= 1

                else:
                    pass
            else:
                for i in range(eig[1]):
                    # Add all the up arrows
                    if electrons_used >= 1:
                        plt.plot(0.2 + i, eig_val, linestyle='none', marker=up_arrow, markersize=15)
                        electrons_used -= 1
                for i in range(eig[1]):
                    # Add all the up arrows
                    if electrons_used >= 1:
                        plt.plot(0.8 + i, eig_val, linestyle='none', marker=down_arrow, markersize=15)
                        electrons_used -= 1

        plt.title('Energy Level Plot for ' + str(self.name))
        plt.xlim(0, max_multiplicity)  # Format the Graph
        plt.xticks([])  # Hide the x-axes
        plt.ylabel('Energy')
        plt.show()

    def find_deloc_energy(self):
        """Calculates the delocalization energy by going through our augmented array and adding up all the energies"""

        deloc_energy = 0.0
        for e, num_electrons in self.e_per_energy_lvl:
            deloc_energy += e * num_electrons
        # We calculated the total Pi Electron Energy

        # Subtract Pi Electrons in Isolated Bonds

        deloc_energy -= (2 * a + 2 * b) * self.num_double_bonds
        if self.alpha is not None and self.beta is not None:
            deloc_energy = deloc_energy.subs(a, self.alpha).subs(b, self.beta)

        self.deloc_energy = smp.N(deloc_energy)  # Set it nicely

        return self.deloc_energy

    def find_charge_density(self):
        """finds the charge density of Pi electrons for each carbon atom in the molecule"""
        charge_density = []

        for c in range(self.num_carbons):  # For each carbon atom
            charge_sum = 0.0
            for eig_index in range(len(self.eigenvectors)):  # For each eigenvector
                num_elec = self.e_per_energy_lvl[eig_index][1]
                charge_sum += num_elec * (self.eigenvectors[eig_index][0][c] ** 2)

            charge_density.append(charge_sum)

        self.charge_density = charge_density  # list of charge densities for carbon atoms 1-n

    def find_bond_order(self):
        """finds the bond order of molecule, stores as list of values beginning at C1"""

        bond_order = []

        for c in range(self.num_carbons - 1 + self.num_additional_connections):
            # For each carbon atom - 1 plus extra bonds between carbons

            bond_sum = 1.0

            for eig_index in range(len(self.eigenvectors)):  # For each eigenvector
                num_elec = self.e_per_energy_lvl[eig_index][1]

                """If the index c goes beyond the number of bonds that would be present in a linear molecule of with
                the same number of carbon atoms, then the bond order for the added connections, i.e bonds found
                outside of the off-diagonals in the Huckel matrix, are calculated"""

                if c < self.num_carbons - 1:
                    bond_sum += num_elec * (self.eigenvectors[eig_index][0][c]) * \
                                self.eigenvectors[eig_index][0][(c + 1)]
                else:
                    bond_sum += num_elec * self.eigenvectors[eig_index][0][
                        self.connections[c % (self.num_carbons - 1)][0] - 1] \
                                * self.eigenvectors[eig_index][0][self.connections[c % (self.num_carbons - 1)][1] - 1]

            bond_order.append(bond_sum)

        self.bond_order = bond_order  # list of charge densities for carbon atoms 1-n

    def normalize_eigenvectors(self):
        """replaces eigenvectors with normalized float type eigenvectors"""

        for eig in self.eigenvectors:
            magnitude = smp.N(eig[0].norm())  # We are using the norm function to calculate the magnitude
            for c in range(self.num_carbons):
                eig[0][c] = float(eig[0][c]) / magnitude

class Carbon:
    def __init__(self, x, y, charge, magnitudes, id):
        self.pos = (x, y)
        self.charge = charge
        self.psi_magnitudes = magnitudes
        self.id = id        # Which carbon atom it is?


class Graphene(Molecule):
    def __init__(self, *args):
        super(Graphene, self).__init__(*args)
        self.carbons = []                           # List of Carbon Atoms

    def generate_carbon(self, id):
        mags = [(abs(eig[id - 1][0]) ** 2) for eig in self.eigenvectors]


armchair = Graphene('Armchair', smp.Matrix([]), 42, 42, 13)
armchair.generate_huckel()
armchair.add_connections([[1, 22], [4, 21], [5, 18], [8, 17], [9, 14], [15, 32], [16, 29], [19, 28],
                          [20, 25], [33, 35], [31, 37], [30, 38], [27, 39], [26, 40], [24, 42]])
armchair.delete_connections([[34, 35], [37, 38], [39, 40]])
print('---', armchair.name, '---')
smp.pprint(armchair.huckel)
armchair.generate_eigen()
print(armchair.eigenvectors)
