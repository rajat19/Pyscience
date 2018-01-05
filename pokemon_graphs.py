import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# from subprocess import check_output
# print(check_output(["ls", "datasets/spotify"]).decode("utf8"))
# f, ax = plt.subplots(figsize=(15, 15))
df = pd.read_csv('datasets/pokemon/pokemon.csv')
small = df.drop(['Name', 'Type 1', 'Type 2'], axis=1)
tips = sns.load_dataset("tips")
ans = sns.load_dataset("anscombe")
attend = sns.load_dataset("attention").query("subject <= 12")
iris = sns.load_dataset("iris")

x = small.Attack
y = small.Defense
tips["weekend"] = tips["day"].isin(["Sat", "Sun"])
tips["big_tip"] = (tips.tip / tips.total_bill) > .175

# sns.distplot(x, rug=True)
# sns.distplot(x, kde=False, bins=20, rug=True)
# sns.distplot(x, hist=False, rug=True)

# sns.kdeplot(x, shade=True, label="HP")
# sns.kdeplot(x, shade=True, cut=0)

# sns.jointplot(x="Attack", y="Defense", data=small)
# sns.jointplot(x=x, y=y, kind="hex", color="k")
# sns.jointplot(x="Attack", y="Defense", data=small, kind="kde")

# sns.kdeplot(x, y, ax=ax)
# sns.rugplot(x, color="g", ax=ax)
# sns.rugplot(y, vertical=True, ax=ax);

# sns.pairplot(small)

# g = sns.pairplot(small)
# g.map_diag(sns.kdeplot)
# g.map_offdiag(sns.kdeplot, cmap="Blues_d", n_levels=6);

# sns.stripplot(x="day", y="total_bill", data=tips)
# sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
# sns.stripplot(x="day", y="total_bill", data=tips, jitter=0.05)
# sns.stripplot(y="day", x="total_bill", data=tips, jitter=True)
# sns.stripplot(y="day", x="total_bill", data=tips, jitter=True, linewidth=1)
# sns.stripplot(x="sex", y="total_bill", hue="day", data=tips, jitter=True)
# sns.stripplot(x="day", y="total_bill", data=tips, jitter=True, hue="smoker", palette="Set2", dodge=True)
# sns.stripplot("day", "total_bill", "smoker", data=tips, palette="Set2", size=20, marker="D", edgecolor="gray", alpha=.25)

# sns.boxplot(x="tip", y="day", data=tips, whis=np.inf)
# sns.stripplot(x="tip", y="day", data=tips, jitter=True, color=".3")

# sns.swarmplot(x="day", y="total_bill", data=tips, hue="sex", palette="Set2", dodge=True, size=6)

# sns.boxplot(x="day", y="total_bill", hue="weekend", data=tips, dodge=True)

# sns.violinplot(x="total_bill", y="day", hue="time", data=tips)
# sns.violinplot(x="total_bill", y="day", hue="time", data=tips, bw=.1, scale="count", scale_hue=False);
# sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True)
# sns.violinplot(x="day", y="total_bill", hue="sex", data=tips, split=True, inner="stick", palette="Set3")

# sns.violinplot(x="day", y="total_bill", data=tips, inner=None)
# sns.swarmplot(x="day", y="total_bill", data=tips, color="w", alpha=.5)

# sns.barplot(x="day", y="total_bill", hue="sex", data=tips, estimator=np.median, ci="sd")
# sns.barplot(x="day", y="tip", data=tips, capsize=.2)
# sns.barplot("size", y="total_bill", data=tips, palette="Blues_d")
# sns.barplot("size", y="total_bill", data=tips, color="salmon", saturation=.5)

# sns.regplot(x="total_bill", y="tip", data=tips, marker="+")
# sns.regplot(x=x, y=y, ci=68, x_jitter=.1)
# sns.regplot(x=x, y=y, x_estimator=np.mean)
# sns.regplot(x="x", y="y", data=ans.loc[ans.dataset == "II"], scatter_kws={"s": 80}, order=2, ci=None, truncate=True)
# sns.regplot(x="total_bill", y="big_tip", data=tips, logistic=True, n_boot=500, y_jitter=.03)

# sns.lmplot(x="total_bill", y="tip", data=tips)
# sns.lmplot(x="size", y="tip", data=tips, x_jitter=.05)
# sns.lmplot(x="total_bill", y="tip", data=tips, lowess=True)
# sns.lmplot(x="total_bill", y="tip", hue="smoker", data=tips, markers=["o", "x"], palette="Set2")

# sns.residplot(x="x", y="y", data=ans.query("dataset == 'II'"), scatter_kws={"s": 80})

# g = sns.FacetGrid(tips, col="day", size=4, aspect=.5)
# g.map(sns.barplot, "sex", "total_bill")

# pal = dict(Lunch="seagreen", Dinner="gray")
# g = sns.FacetGrid(tips, hue="time", palette=pal, size=5)
# g.map(plt.scatter, "total_bill", "tip", s=50, alpha=.7, linewidth=.5, edgecolor="white")
# g.add_legend()

# g = sns.FacetGrid(tips, hue="sex", palette="Set1", size=5, hue_kws={"marker": ["^", "v"]})
# g.map(plt.scatter, "total_bill", "tip", s=100, linewidth=.5, edgecolor="white")
# g.add_legend()

# g = sns.FacetGrid(attend, col="subject", col_wrap=4, size=2, ylim=(0, 10))
# g.map(sns.pointplot, "solutions", "score", color=".3", ci=None)

# with sns.axes_style("white"):
#     g = sns.FacetGrid(tips, row="sex", col="smoker", margin_titles=True, size=2.5)
# g.map(plt.scatter, "total_bill", "tip", color="#334488", edgecolor="white", lw=.5)
# g.set_axis_labels("Total bill (US Dollars)", "Tip");
# g.set(xticks=[10, 30, 50], yticks=[2, 6, 10])
# g.fig.subplots_adjust(wspace=.02, hspace=.02)

# def hexbin(x, y, color, **kwargs):
#     cmap = sns.light_palette(color, as_cmap=True)
#     plt.hexbin(x, y, gridsize=15, cmap=cmap, **kwargs)

# with sns.axes_style("dark"):
#     g = sns.FacetGrid(tips, hue="time", col="time", size=4)
# g.map(hexbin, "total_bill", "tip", extent=[0, 50, 0, 10])

# g = sns.PairGrid(iris, hue="species")
# g.map_diag(plt.hist)
# g.map_offdiag(plt.scatter)
# g.add_legend();

# g = sns.PairGrid(iris)
# g.map_upper(plt.scatter)
# g.map_lower(sns.kdeplot, cmap="Blues_d")
# g.map_diag(sns.kdeplot, lw=3, legend=False)

g = sns.pairplot(iris, hue="species", palette="Set2", diag_kind="kde", size=2.5)

plt.show()