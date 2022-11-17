from molmass import ELEMENTS, Element
from os import listdir, path
import pandas as pd
from tkinter import filedialog


def calc_com(m1, c1, m2, c2):
	return (m1 * c1 + m2 * c2) / (m1 + m2)


def process_xyz():
	xyz_list = listdir(structures_dir)  # List of .xyz files

	# Iterate through .xyz list
	for f in xyz_list[1:2]:
		# Molecule identifier
		mol_id = path.splitext(path.basename(structures_dir + f))[0]

		# DF of current .xyz file
		current_xyz = pd.read_csv(structures_dir + f, skiprows=2, header=None, sep=' ')

		# Rows that match the current .xyz file molecule
		matching_atom_pairs = train_df[train_df['molecule_name'] == mol_id].loc[:, ['atom_index_0', 'atom_index_1']]

		# Iterate through atomic pairs from matching_atom_rows
		for atom_pair in matching_atom_pairs.iterrows():  # index 0 is 'index', index 1 is Pandas Series of the two indexes
			# Row of corresponding atoms in pair
			atom_1 = atom_pair[1]['atom_index_0']
			atom_2 = atom_pair[1]['atom_index_1']

			# Sets to atomic symbol
			atom_1_type = current_xyz.iloc[atom_1, 0]
			atom_2_type = current_xyz.iloc[(atom_2, 0)]

			# Pandas Series; index 1 is x, 2 is y, 3 is z coordinate
			atom_1_coords = current_xyz.iloc[atom_1, 1:4]
			atom_2_coords = current_xyz.iloc[atom_2, 1:4]

			# Calculate Center of Mass for the atom pair
			com = calc_com(ELEMENTS[atom_1_type].mass, atom_1_coords, ELEMENTS[atom_2_type].mass, atom_2_coords)

			distances = {}

			index_list = list(range(0, current_xyz.shape[0]))
			index_list.remove(atom_1)
			index_list.remove(atom_2)

			for atom in current_xyz.iloc[index_list].iterrows():
				pair_com = calc_com((ELEMENTS[atom_1_type].mass + ELEMENTS[atom_2_type].mass)/2, com, ELEMENTS[atom[1][0]].mass, atom[1][1:4])


if __name__ == "__main__":
	# TODO: Uncomment tkinter dialogs; delete hard coded directories
	# train_file = filedialog.askopenfilename(title='Select Training File', initialdir='./')
	# structures_dir = filedialog.askdirectory(title='Select Structures Directory', initialdir='./')
	train_file = './data/train.csv'
	structures_dir = './data/structures/'
	train_df = pd.read_csv(train_file)
	train_df['n1_mol'] = ''
	train_df['n1_dist'] = 0
	train_df['n2_mol'] = ''
	train_df['n2_dist'] = 0
	process_xyz()
