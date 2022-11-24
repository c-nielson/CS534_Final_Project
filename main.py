import concurrent.futures
import csv
import functools
import itertools
import math
from os import listdir, path
from tkinter import filedialog, simpledialog

import pandas
import pandas as pd
from molmass import ELEMENTS


def calc_com(m1: float, c1: pandas.Series, m2: float, c2: pandas.Series) -> pandas.Series:
	"""
	Calculates the center of mass between the given masses and their coordinate tuples.

	:param m1: Mass 1
	:type m1:
	:param c1: Coordinates of m1
	:type c1:
	:param m2: Mass 2
	:type m2:
	:param c2: Coordinates of m2
	:type c2:
	:return: Center of mass coordinates
	:rtype: tuple(float)
	"""
	return (m1 * c1 + m2 * c2) / (m1 + m2)


def process_xyz(
	train_df: pandas.DataFrame, xyz_file: str, file_counter=None, total_files: int = None
) -> list:
	"""
	Processes one .xyz file; calculates centers of mass of atom combinations and finds nearest neighbors. Writes result using csv_out.

	:param train_df: DataFrame containing all training points
	:type train_df:
	:param xyz_file: .xyz file to process
	:type xyz_file:
	:param file_counter: current number of file; for console writing purposes only
	:type file_counter: iterator
	:param total_files: total number of files; for console writing purposes only
	:type total_files:
	:return: list of results
	:rtype: list
	"""
	# Molecule identifier
	mol_id = path.splitext(path.basename(xyz_file))[0]

	# DF of current .xyz file
	current_xyz = pd.read_csv(xyz_file, skiprows=2, header=None, sep=' ')

	# Rows that match the current .xyz file molecule
	matching_atom_pairs = train_df[train_df['molecule_name'] == mol_id]

	results = []

	# Iterate through atomic pairs from matching_atom_pairs
	for (_, row) in matching_atom_pairs.iterrows():
		# Row of corresponding atoms in pair
		atom_0_index = row['atom_index_0']
		atom_1_index = row['atom_index_1']

		# Sets to atomic symbol
		atom_0_type = current_xyz.iloc[atom_0_index, 0]
		atom_1_type = current_xyz.iloc[atom_1_index, 0]

		# Pandas Series; index 1 is x, 2 is y, 3 is z coordinate
		atom_0_coords = current_xyz.iloc[atom_0_index, 1:4]
		atom_1_coords = current_xyz.iloc[atom_1_index, 1:4]

		# Calculate Center of Mass for the atom pair
		com = calc_com(ELEMENTS[atom_0_type].mass, atom_0_coords, ELEMENTS[atom_1_type].mass, atom_1_coords)

		# Dictionary of atom index and (mass-weighted) distance to atomic pair COM
		distances = {}

		# List of atom indices excluding the current atom pair
		index_list = filter(
			lambda i: i != atom_0_index and i != atom_1_index, range(0, current_xyz.shape[0])
		)

		# Iterate through index_list and calculate (mass-weighted) COM between atom and current pair COM
		for atom in current_xyz.iloc[index_list].iterrows():
			# Center of mass between atomic pair COM and atom
			pair_com = calc_com((ELEMENTS[atom_0_type].mass + ELEMENTS[atom_1_type].mass) / 2, com, ELEMENTS[atom[1][0]].mass, atom[1][1:4])
			distances[atom[0]] = math.sqrt(functools.reduce(lambda x, y: x + y, (com.array - pair_com) ** 2))

		# Sort by distance and calculate closest 2
		sorted_distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}
		closest_2 = tuple(itertools.islice(sorted_distances.items(), 2))

		results.append(
			[row['id'], mol_id, atom_0_index, atom_0_type, atom_1_index, atom_1_type, row['type'], row['scalar_coupling_constant'],
				current_xyz.iloc[closest_2[0][0]][0], closest_2[0][1], current_xyz.iloc[closest_2[1][0]][0] if len(closest_2) > 1 else '',
				closest_2[1][1] if len(closest_2) > 1 else 0]
		)

	if file_counter is not None and total_files is not None:
		print(f'\tFinished {next(file_counter)} / {total_files}:\n\t\t{xyz_file}')

	return results


def main() -> None:
	"""
	Main entrypoint to program. Sets up the input and output files and spins off threads to handle each .xyz file.
	:return: None
	:rtype: None
	"""
	# Open dialogs to select input and output files, and number of threads to run on
	train_file = filedialog.askopenfilename(title='Select Training File', initialdir='./')
	structures_dir = filedialog.askdirectory(title='Select Structures Directory', initialdir='./')
	output_file = filedialog.asksaveasfilename(title='Select Save File', initialdir='./', filetypes=[('CSV', '*.csv')])
	num_threads = simpledialog.askinteger(title='Number of Threads', prompt='How many threads do you want to use?')

	xyz_list = listdir(structures_dir)
	total_files = len(xyz_list)
	num_files = iter(range(1, total_files + 1))

	train_df = pd.read_csv(train_file)

	# Spin off threads
	print('Processing files...')
	with concurrent.futures.ThreadPoolExecutor(num_threads) as executor:
		futures = [executor.submit(
			process_xyz, train_df, structures_dir + '/' + file, num_files, total_files
		) for file in xyz_list]

		# Wait for all .xyz files to be processed
		concurrent.futures.wait(futures)

	futures_total = len(futures)
	futures_num = iter(range(1, futures_total + 1))
	print('\nWriting results...')
	# Write all results to file
	with open(output_file, 'w', newline='') as csv_file:
		csv_out = csv.writer(csv_file)
		# Write header
		csv_out.writerow(
			['id', 'molecule_name', 'atom_index_0', 'atom_type_0', 'atom_index_1', 'atom_type_1', 'type', 'scalar_coupling_constant', 'n1_type',
				'n1_dist', 'n2_type', 'n2_dist']
		)

		for future in futures:
			print(f'\t{next(futures_num)} / {futures_total}')
			for result in future.result():
				csv_out.writerow(result)

	print('\nFinished!')


if __name__ == "__main__":
	import sys

	sys.exit(main())
