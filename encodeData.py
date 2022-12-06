from tkinter import filedialog

import pandas as pd
from sklearn import preprocessing


def main():
	data_in = filedialog.askopenfilename(title='Select File for Encoding', initialdir='./data/')
	data_out = filedialog.asksaveasfilename(title='Select File to Save Results', initialdir='./data/', filetypes=[('CSV', '*.csv')])

	cols_to_encode = [
		'atom_type_0',
		'atom_type_1',
		'type',
		'n1_type',
		'n2_type',
		'n3_type',
		'n4_type',
		'n5_type',
		'n6_type',
		'n7_type',
		'n8_type',
		'n9_type',
		'n10_type'
	]

	df = pd.read_csv(
		data_in,
		usecols=cols_to_encode
	)

	df_encoded = pd.read_csv(
		data_in,
		usecols=[
			'pair_dist',
			'n1_dist',
			'n2_dist',
			'n3_dist',
			'n4_dist',
			'n5_dist',
			'n6_dist',
			'n7_dist',
			'n8_dist',
			'n9_dist',
			'n10_dist',
			'scalar_coupling_constant'
		]
	)

	for col in cols_to_encode:
		df_encoded = df_encoded.join(
			pd.get_dummies(df[[col]])
		)

	max_abs_scaler = preprocessing.MaxAbsScaler()
	df_encoded[['scalar_coupling_constant']] = max_abs_scaler.fit_transform(df_encoded[['scalar_coupling_constant']])

	df_encoded.to_csv(data_out)


if __name__ == "__main__":
	import sys

	sys.exit(main())
