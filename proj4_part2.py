"""PART 2"""

from proj4_v1 import Molecule, a, b
import sympy as smp

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
        """Mint a new Carbon atom"""
        mags = [(abs(eig[id - 1][0]) ** 2) for eig in self.eigenvectors]

        def find_position(self):
            """Finds the x, y coord of a specific carbon atom"""
            pass


"""Part 2: Armchair Graphene"""
if __name__ == '__main__':
    # Armchair
    armchair = Graphene('Armchair', smp.Matrix([]), 42, 42, 13)
    armchair.set_constants(0, 1)
    armchair.generate_huckel()
    armchair.add_connections([[1, 22], [4, 21], [5, 18], [8, 17], [9, 14], [15, 32], [16, 29], [19, 28],
                              [20, 25], [33, 35], [31, 37], [30, 38], [27, 39], [26, 40], [24, 42]])
    armchair.delete_connections([[34, 35], [37, 38], [39, 40]])
    print('---', armchair.name, '---')
    smp.pprint(armchair.huckel)
    armchair.generate_eigen()
    print('ev', armchair.eigenvalues)
    print('ev', armchair.eigenvectors)


    # Zigzag
    zigzag = Graphene('ZigZag', smp.Matrix([]), 36, 36, 18)
    armchair.set_constants(0, 1)
    zigzag.generate_huckel()
    zigzag.add_connections([[1, 18], [4, 17], [6, 15], [8, 13], [12, 25], [14, 23], [16, 21], [20, 33],
                            [22, 31], [24, 29], [36, 19]])
    print('---', zigzag.name, '---')
    smp.pprint(zigzag.huckel)
    zigzag.generate_eigen()
    print('ev', zigzag.eigenvalues)
    print('ev', zigzag.eigenvectors)