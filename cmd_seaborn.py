# -*- coding: utf-8 -*-

%matplotlib inline

import numpy as np
import pandas as pd
from scipy import stats, integrate
import matplotlib as mpl
import matplotlib.pyplot as plt

import seaborn as sns
sns.set(color_codes=True)

np.random.seed(sum(map(ord, "distributions")))

########################### distribution ###################################
x = np.random.normal(size=100)
sns.distplot(x)
sns.distplot(x,bins=20)

# bivariate distribution
mean, cov = [0, 1], [(1, .5), (.5, 1)]
data = np.random.multivariate_normal(mean, cov, 200)
df = pd.DataFrame(data, columns=["x", "y"])

sns.jointplot(x="x", y="y", data=df);   # scatter plot
sns.jointplot(x="x", y="y", data=df, kind="hex", color="k");    # hex bin plot
sns.jointplot(x="x", y="y", data=df, kind="kde");   # contour plot

# multivariate scatter plot. ref: http://seaborn.pydata.org/generated/seaborn.pairplot.html#seaborn.pairplot
iris = sns.load_dataset("iris")
sns.pairplot(iris);
sns.pairplot(iris,hue='species')
sns.pairplot(iris, hue="species", palette="husl")
sns.pairplot(iris, hue="species", markers=["o", "s", "D"])
sns.pairplot(iris, vars=["sepal_width", "sepal_length"],hue='species')


##################### Catagorical plots #######################
# ref: http://seaborn.pydata.org/tutorial/categorical.html
sns.set(style="whitegrid", color_codes=True)
np.random.seed(sum(map(ord, "categorical")))

titanic = sns.load_dataset("titanic")
tips = sns.load_dataset("tips")
iris = sns.load_dataset("iris")

# box plot
sns.boxplot(data=tips, x="day", y="total_bill");
# bar plot
sns.barplot(data=titanic, x="sex", y="survived" );
sns.barplot(data=titanic, x="sex", y="survived", hue="class");
# count plot
sns.countplot(data=titanic, y="deck", color="c");
# factor plot. ref: http://seaborn.pydata.org/generated/seaborn.factorplot.html#seaborn.factorplot
sns.factorplot(data=tips, x="day", y="total_bill",col="time", kind="box");
sns.factorplot(data=tips, x="day", y="total_bill", row="sex", col="time", kind="box");
sns.factorplot(data=tips, x="day", y="total_bill", row="sex", col="time");
