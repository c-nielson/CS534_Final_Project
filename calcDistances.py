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


def get_dist(a, b) -> float:
	"""
	Returns distance between two sets of coordinates.

	:param a: First coordinate
	:type a: Array-like
	:param b: Second coordinate
	:type b: Array-like
	:return: Distance between a and b
	:rtype: float
	"""
	return math.sqrt(functools.reduce(lambda x, y: x + y, (a - b) ** 2))


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
			distances[atom[0]] = get_dist(com.array, pair_com)

		# Sort by distance and calculate closest 2
		sorted_distances = {k: v for k, v in sorted(distances.items(), key=lambda item: item[1])}

		result_array = [
			row['id'],
			mol_id,
			atom_0_index,
			atom_0_type, 
			ELEMENTS[atom_0_type].number,
			atom_1_index, 
			atom_1_type,
			ELEMENTS[atom_1_type].number, 
			get_dist(atom_0_coords.array, atom_1_coords.array), 
			row['type'],
			row['scalar_coupling_constant']
		]

		# Add all distances calculated (closest first)
		for neighbour in sorted_distances.items():
			neighbour_type = current_xyz.iloc[neighbour[0]][0],
			result_array.extend([
				neighbour_type[0],
				ELEMENTS[neighbour_type[0]].number,
				neighbour[1]
			])

		# Add blank entries as necessary to have 10 neighbours total
		result_array.extend(['', 0, 0] * (10 - len(sorted_distances)))

		# Add to results list
		results.append(result_array)

	# Just for logging to track progress
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
			[
				'id',
				'molecule_name',
				'atom_index_0',
				'atom_type_0',
				'atomic_number_0',
				'atom_index_1',
				'atom_type_1',
				'atomic_number_1',
				'pair_dist',
				'type',
				'scalar_coupling_constant',
				'n1_type',
				'n1_number',
				'n1_dist',
				'n2_type',
				'n2_number',
				'n2_dist',
				'n3_type',
				'n3_number',
				'n3_dist',
				'n4_type',
				'n4_number',
				'n4_dist',
				'n5_type',
				'n5_number',
				'n5_dist',
				'n6_type',
				'n6_number',
				'n6_dist',
				'n7_type',
				'n7_number',
				'n7_dist',
				'n8_type',
				'n8_number',
				'n8_dist',
				'n9_type',
				'n9_number',
				'n9_dist',
				'n10_type',
				'n10_number',
				'n10_dist'
			]
		)

		# Write results of all Futures
		for future in futures:
			print(f'\t{next(futures_num)} / {futures_total}')
			for result in future.result():
				csv_out.writerow(result)

	print('\nFinished!')


# Entry point
if __name__ == "__main__":
	import sys

	sys.exit(main())