import concurrent.futures
import functools
import itertools
import math
from os import listdir, path
from tkinter import filedialog

import pandas as pd
from molmass import ELEMENTS


def calc_com(m1, c1, m2, c2):
	return (m1 * c1 + m2 * c2) / (m1 + m2)


def process_xyz(train_df, xyz_file, file_num, total_files):
	print(f'Processing file {file_num} / {total_files}:\n\t{xyz_file}')

	# Molecule identifier
	mol_id = path.splitext(path.basename(xyz_file))[0]

	# DF of current .xyz file
	current_xyz = pd.read_csv(xyz_file, skiprows=2, header=None, sep=' ')

	# Rows that match the current .xyz file molecule
	matching_atom_pairs = train_df[train_df['molecule_name'] == mol_id]

	result = [train_df]

	# Iterate through atomic pairs from matching_atom_pairs
	for (_, row) in matching_atom_pairs.iterrows():
		# Row of corresponding atoms in pair
		atom_1_index = row['atom_index_0']
		atom_2_index = row['atom_index_1']

		# Sets to atomic symbol
		atom_1_type = current_xyz.iloc[atom_1_index, 0]
		atom_2_type = current_xyz.iloc[atom_2_index, 0]

		# Pandas Series; index 1 is x, 2 is y, 3 is z coordinate
		atom_1_coords = current_xyz.iloc[atom_1_index, 1:4]
		atom_2_coords = current_xyz.iloc[atom_2_index, 1:4]

		# Calculate Center of Mass for the atom pair
		com = calc_com(ELEMENTS[atom_1_type].mass, atom_1_coords, ELEMENTS[atom_2_type].mass, atom_2_coords)

		# Dictionary of atom index and (mass-weighted) distance to atomic pair COM
		distances = {}

		# List of atom indices excluding the current atom pair
		index_list = filter(
			lambda i: i != atom_1_index and i != atom_2_index, range(0, current_xyz.shape[0])
		)

		# Iterate through index_list and calculate (mass-weighted) COM between atom and current pair COM
		for atom in current_xyz.iloc[index_list].iterrows():
			# Center of mass between atomic pair COM and atom
			pair_com = calc_com((ELEMENTS[atom_1_type].mass + ELEMENTS[atom_2_type].mass) / 2, com, ELEMENTS[atom[1][0]].mass, atom[1][1:4])
			distances[atom[0]] = math.sqrt(functools.reduce(lambda x, y: x + y, (com.array - pair_com) ** 2))

		# Sort by distance and calculate closest 2
		sorted_distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
		closest_2 = tuple(itertools.islice(sorted_distances.items(), 2))

		result.append(
			{
				'index': row['id'],
				'atom_type_0': atom_1_type,
				'atom_type_1': atom_2_type,
				'n1_mol': current_xyz.iloc[closest_2[0][0]][0],
				'n1_dist': closest_2[0][1],
				'n2_mol': current_xyz.iloc[closest_2[1][0]][0] if len(closest_2) > 1 else '',
				'n2_dist': closest_2[1][1] if len(closest_2) > 1 else 0
			}
		)

	return result


def update_df(future: concurrent.futures.Future):
	train_df = future.result()[0]

	for result in future.result()[1:]:
		train_df.loc[result['index'], 'n1_mol'] = result['n1_mol']
		train_df.loc[result['index'], 'n1_dist'] = result['n1_dist']

		train_df.loc[result['index'], 'n2_mol'] = result['n2_mol']
		train_df.loc[result['index'], 'n2_dist'] = result['n2_dist']

		train_df.loc[result['index'], 'atom_type_0'] = result['atom_type_0']
		train_df.loc[result['index'], 'atom_type_1'] = result['atom_type_1']


def main():
	train_file = filedialog.askopenfilename(title='Select Training File', initialdir='./')
	structures_dir = filedialog.askdirectory(title='Select Structures Directory', initialdir='./')

	xyz_list = listdir(structures_dir)  # List of .xyz files
	total_files = len(xyz_list)
	num_files = iter(range(1, total_files + 1))

	train_df = pd.read_csv(train_file)

	train_df['atom_type_0'] = ''
	train_df['atom_type_1'] = ''
	train_df['n1_mol'] = ''
	train_df['n1_dist'] = 0
	train_df['n2_mol'] = ''
	train_df['n2_dist'] = 0

	with concurrent.futures.ThreadPoolExecutor(4) as executor:
		futures = [executor.submit(
			process_xyz, train_df, structures_dir + '/' + file, next(num_files), total_files
		) for file in xyz_list[:3]]
		for future in futures:
			future.add_done_callback(update_df)
		concurrent.futures.wait(futures)  # executor.shutdown()

	print('Saving...')
	train_df.to_csv('./data/train_w_knn.csv')
	print('Finished!')


if __name__ == "__main__":
	import sys

	sys.exit(main())
