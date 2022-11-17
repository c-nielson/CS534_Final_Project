from molmass import ELEMENTS, Element
from os import listdir, path
import pandas as pd
from tkinter import filedialog


def process_xyz():
	xyz_list = listdir(structures_dir)
	for f in xyz_list[1:2]:
		mol_id = path.splitext(path.basename(structures_dir + f))[0]
		current_xyz = pd.read_csv(structures_dir + f, skiprows=2, header=None, sep=' ')
		matching_atom_pairs = train_df[train_df['molecule_name'] == mol_id].loc[:, ['atom_index_0', 'atom_index_1']]
		for atom_pair in matching_atom_pairs.iterrows():  # index 0 is 'index', index 1 is Pandas Series of the two indexes
			atom_1 = atom_pair[1]['atom_index_0']
			atom_2 = atom_pair[1]['atom_index_1']

			# Sets to atomic symbol
			atom_1_type = current_xyz.iloc[atom_1, 0]
			atom_2_type = current_xyz.iloc[(atom_2, 0)]

			# Pandas Series; index 1 is x, 2 is y, 3 is z coordinate
			atom_1_coords = current_xyz.iloc[atom_1, 1:4]
			atom_2_coords = current_xyz.iloc[atom_2, 1:4]

			# Calculate Center of Mass for the atom pair
			com = (ELEMENTS[atom_1_type].mass * atom_1_coords + ELEMENTS[atom_2_type].mass * atom_2_coords) / (
					ELEMENTS[atom_1_type].mass + ELEMENTS[atom_2_type].mass)


if __name__ == "__main__":
	# train_file = filedialog.askopenfilename(title='Select Training File', initialdir='./')
	# structures_dir = filedialog.askdirectory(title='Select Structures Directory', initialdir='./')
	train_file = './data/train.csv'
	structures_dir = './data/structures/'
	train_df = pd.read_csv(train_file)
	process_xyz()
