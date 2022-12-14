import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt


def main():
	data = {
		'1JHN': [-0.9549, -0.1255],
		'1JHC': [-0.2505, 0.5465],
		'2JHH': [-1.7167, -1.1283],
		'2JHN': [-1.9253, -0.7857],
		'2JHC': [-1.1799, 0.0700],
		'3JHH': [-1.7352, -1.1085],
		'3JHC': [-1.0811, 0.0176],
		'3JHN': [-2.171047, -1.1294],
		'Method': ['Original', 'Mass Weighted']
	}

	df = pd.DataFrame(data)
	melted = df.melt(id_vars='Method', var_name='Coupling Type', value_name='log(MAE)')

	sns.set(rc={'figure.figsize': (11, 8.5)})
	sns.barplot(data=melted, x='Coupling Type', y='log(MAE)', hue='Method').set(title='Mass-Weighted vs Original Scheme')
	plt.savefig('kaggle_comparison.png')
	plt.show()

if __name__ == '__main__':
	import sys
	sys.exit(main())
