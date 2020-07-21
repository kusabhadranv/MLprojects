# Load libraries
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC

# Load dataset
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"

#Add column names to the dataset
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Check the details of the dataset. It has 150 instances with 5 attributes
#print(dataset.shape)

# Print first 20 rows of the dataset
#print(dataset.head(20))

# Get the details of each of the attribute. Count, Mean, Std, etc
#print(dataset.describe())

# Check the class distribution. Each of the class has same number of examples - 50.
#print(dataset.groupby('class').size())

# Univariate plot - Plot for each variable(columns). This is a box and whisker plot.
#dataset.plot(kind='box', subplots=True, layout=(2,2), sharex=False, sharey=False)
#pyplot.show()

# Another univariate plot. This is histogram.
#dataset.hist()
#pyplot.show()

# scatter plot matrix
scatter_matrix(dataset)
pyplot.show()
