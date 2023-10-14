import pandas as pd
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import seaborn as sns

penguins_data = pd.read_csv('penguins.csv')

species = {
    "Adelie": 0,
    "Gentoo": 1,
    "Chinstrap": 2
}

islands = {
    "Biscoe": 0,
    "Dream": 1,
    "Torgersen": 2
}

sex = {
    "MALE": 0,
    "FEMALE": 1
}

dataframe = None

for specie in species:
    print("Nome da especie:", specie)
    df = penguins_data[(penguins_data["species"] == specie)]
    df = df.dropna().sample(n=60, random_state=42)

    df["species_number"] = df["species"].map(species)
    df["island_number"] = df["island"].map(islands)
    df["sex_number"] = df["sex"].map(sex)

    if dataframe is None:
        dataframe = df
    else:
        dataframe = pd.concat([dataframe, df])

clf = DecisionTreeClassifier(criterion="entropy")

attributes = dataframe.drop(['species', 'species_number', 'sex', 'island'], axis=1, inplace=False)
labels = dataframe['species_number'].copy()

clf.fit(attributes, labels)

fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(10,10))
tree.plot_tree(clf, feature_names=attributes.columns, class_names={0: "Adelie", 1: "Gentoo", 2: "Chinstrap"}, ax=ax,
               filled=True)

plt.show()


def plot(x, y, data, xlabel, ylabel):
    sns.barplot(x=x, y=y, data=data)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title('Relação entre ' + xlabel + ' e ' + ylabel)

    plt.show()


plot("species", "flipper_length_mm", dataframe, "Espécie", "Comprimento da Nadadeira")
plot("species", "body_mass_g", dataframe, "Espécie", "Peso")
plot("species", "bill_length_mm", dataframe, "Espécie", "Comprimento ")
plot("sex", "bill_length_mm", dataframe, "Sexo", "Comprimento do Bico")
plot("sex", "body_mass_g", dataframe, "Sexo", "Peso")
