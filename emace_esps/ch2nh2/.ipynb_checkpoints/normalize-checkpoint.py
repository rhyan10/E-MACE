import ase.io
import numpy as np
from ase import Atoms

db = ase.io.read("ch2nh2_f_adjusted.xyz", ":")

energies = []
for i in range(len(db)):
    energies.append(db[i].info["energy"])

energies = np.array(energies)
means = np.mean(energies, axis=0)

edited_db = []
for i in range(len(db)):
    energy = db[i].info["energy"] - means
    forces = db[i].info["forces"]
    positions = db[i].positions
    symbols = db[i].numbers
    mol = Atoms(positions = positions, numbers = symbols)
    mol.info = {"energy": energy, "forces": forces}
    edited_db.append(mol)

ase.io.write("ch2nh2_f_adjusted_mean.xyz", edited_db)