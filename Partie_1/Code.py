# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
###______Question 2.2____________________ 


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Je charge mon fichier excel
xls = pd.ExcelFile("h:\Mes documents\RES_all_cleaned.xlsx")

# J'affiche les noms des feuilles constituant le fichier 
print(xls.sheet_names)

# Je charge ma feuille TD 
df = xls.parse("TD")

# J'affiche le nombre de valeurs manquantes par colonne
print(df.isnull().sum())

# Affiche les lignes contenant des valeurs manquantes
print(df[df.isnull().any(axis=1)])

# Remplace les valeurs manquantes par la médiane de chaque colonne
df_cleaned = df.fillna(df.median(numeric_only=True))

# Remarquons que nous retrouvons toujours des valeurs manquantes donc 
# on va supprimer les colonnes entièrement vides 
df_cleaned = df_cleaned.dropna(axis=1, how='all')
print(df_cleaned.isnull().sum())

# Nous allons supprimer les colonnes AWR,COG,COM,MOT,RRB,SCI,TOT car à la base ces colonnes ne possédaient que trois valeurs sur normalement 30 valeurs 
# Ce qui ne nous permet pas de bien avancer dans notre cas d'étude de pas car dans une expérience où l'on se base que sur le résultat de seulement 03 Individus
# par rapport à 30 personnes aucune dédudction concrète n'en ressortira
cols_to_drop = ['AWR', 'COG', 'COM', 'MOT', 'RRB', 'SCI', 'TOT']
df_cleaned = df_cleaned.drop(columns=cols_to_drop)

# Fonction pour détecter les outliers
def detect_outliers(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

# D'abord nous sélectionnons uniquement les colonnes numériques
numeric_cols = [col for col in df_cleaned.columns if df_cleaned[col].dtype in ['float64', 'int64']]

# Nous donnons le nombre de boxplots à afficher par figure
boxplots_per_figure = 6  

# Nous écrivons le code pour qu'il nous affiche les boxplots par groupes
for i in range(0, len(numeric_cols), boxplots_per_figure):
    fig, axes = plt.subplots(1, boxplots_per_figure, figsize=(20, 5))
    
    for j, col in enumerate(numeric_cols[i:i + boxplots_per_figure]):
        sns.boxplot(y=df_cleaned[col], ax=axes[j])
        axes[j].set_title(f'Boxplot de {col}')
    plt.tight_layout()
    plt.show(block=True)

# Remplacer les outliers par la médiane
for col in numeric_cols:
    outliers = detect_outliers(df_cleaned, col)
    median_value = df_cleaned[col].median()
    df_cleaned[col] = np.where((df_cleaned[col] < (df_cleaned[col].quantile(0.25) - 1.5 * (df_cleaned[col].quantile(0.75) - df_cleaned[col].quantile(0.25)))) | 
                               (df_cleaned[col] > (df_cleaned[col].quantile(0.75) + 1.5 * (df_cleaned[col].quantile(0.75) - df_cleaned[col].quantile(0.25)))), 
                               median_value, df_cleaned[col])

# Vérification après remplacement
print(df_cleaned.isnull().sum())




#### _______Question 2.3______________

# Ici nous nommons df_desc le tableau pour synthétiser 
# nos différentes données respectives . 

df_desc = df_cleaned.describe()


# Nous passons ici dans ce cas à la représentation graphique de chaque variable .Ici le choix des graphiques
# qu'on a voulu utiliser pour chaque variable dépend des natures de leur valeurs numériques 


import matplotlib.pyplot as plt
import seaborn as sns

# Liste des variables continues (ms, m)

variables_continues = [
    "TAju(ms)", "TExe(ms)", "TPas(ms)", "TLoad(ms)", "TUnload(ms)", 
    "BOS_AP_Init(m)", "BOS_ML_Init(m)"
]

# Liste des variables normalisées (/Long, /Large)
variables_normalisees = [
    "CoP_tot_AP(/Long)", "CoP_tot_ML(/Larg)", "MOS_AP_FOcr(/Long)", "MOS_ML_FOcr(/Larg)", 
    "CoP_loadAP(/Long)", "CoP_load_ML(/Larg)", "CoP_unload_ML(/Larg)"
]

variables_vitesses = [
    "Vpic_load_AP(m/s)", "Vpic_load_ML(m/s)", "Vpic_unload_ML(m/s)"
]

# Définir la taille de la figure pour tout afficher
plt.figure(figsize=(18, 12))

# Tracer les histogrammes pour les variables continues (ms,m)
for i, var in enumerate(variables_continues):
    plt.subplot(4, 5, i+1)
    sns.histplot(df_cleaned[var], kde=True)
    plt.title(f"Histogramme de {var}")

# Tracer les boxplots pour les variables normalisées (/Long, /Large)
for j, var in enumerate(variables_normalisees, start=len(variables_continues)):
    plt.subplot(4, 5, j+1)
    sns.boxplot(x=df_cleaned[var])
    plt.title(f"Boxplot de {var}")
    
    
    
# Tracer les violons pour les variables vitesses (m/s)
for k, var in enumerate(variables_vitesses, start=len(variables_continues) + len(variables_normalisees)):
    plt.subplot(4, 5, k+1)
    sns.violinplot(x=df_cleaned[var])
    plt.title(f"Violin Plot de {var}")

# Ajuster l'affichage des sous-graphes
plt.tight_layout()
plt.show()



######_____________________________________________________________________________________________
######_____________________________________________________________________________________________

######_________________Question 3__________________________________________________________________

######_________________Question 3.1_________________

import pandas as pd

# Fonction pour catégoriser les âges
def categorize_age(age):
    if 6 <= age < 9:  
        return "6-8 ans"
    elif 9 <= age < 12 : 
        return "9-11 ans"
    else:
        return None  

# Ajouter la colonne "AGE_GROUPE_2"
df_cleaned["AGE_GROUPE_2"] = df_cleaned["Age"].apply(categorize_age)

# Vérifier la répartition
print(df_cleaned["AGE_GROUPE_2"].value_counts())

# Tout d'abord pour proposer un test statistique pour comparer ses deux groupes d'enfant il nous faut tout d'abord 
# vérifier que la nature normale ou anormale de nos valeurs . Sur ce nous allons apliquer Shapiro-Wilk pour tester la normalité 
# des valeurs et Levene pour tester l'homohgénéité des variances des variables

import pandas as pd
from scipy.stats import shapiro, levene, ttest_ind, mannwhitneyu

variables = ["TAju(ms)", "TExe(ms)", "TPas(ms)", "TLoad(ms)", "TUnload(ms)", 
"BOS_AP_Init(m)", "BOS_ML_Init(m)","CoP_tot_AP(/Long)", "CoP_tot_ML(/Larg)", "MOS_AP_FOcr(/Long)", "MOS_ML_FOcr(/Larg)", 
"CoP_loadAP(/Long)", "CoP_load_ML(/Larg)", "CoP_unload_ML(/Larg)",    "Vpic_load_AP(m/s)", "Vpic_load_ML(m/s)", "Vpic_unload_ML(m/s)"
]


# Dictionnaire pour stocker les résultats
results = {
    "Variable": [],
    "Test utilisé": [],
    "p (Shapiro 6-8 ans)": [],
    "p (Shapiro 9-11 ans)": [],
    "p (Levene)": [],
    "p (Test Statistique)": []
}

# Application des tests pour chaque variable
for var in variables:
    group1 = df_cleaned[df_cleaned["AGE_GROUPE_2"] == "6-8 ans"][var]
    group2 = df_cleaned[df_cleaned["AGE_GROUPE_2"] == "9-11 ans"][var]

    # Test de normalité
    stat1, p1 = shapiro(group1)
    stat2, p2 = shapiro(group2)

    # Test d'homogénéité des variances
    stat_lev, p_lev = levene(group1, group2)

    # Choisir le test statistique approprié
    if p1 > 0.05 and p2 > 0.05 and p_lev > 0.05:
        stat, p_test = ttest_ind(group1, group2, equal_var=True)
        test_name = "Test t de Student"
    else:
        stat, p_test = mannwhitneyu(group1, group2)
        test_name = "Test de Mann-Whitney"

    # Ajouter les résultats dans le dictionnaire
    results["Variable"].append(var)
    results["Test utilisé"].append(test_name)
    results["p (Shapiro 6-8 ans)"].append(p1)
    results["p (Shapiro 9-11 ans)"].append(p2)
    results["p (Levene)"].append(p_lev)
    results["p (Test Statistique)"].append(p_test)

# Créer le DataFrame
results_df = pd.DataFrame(results)

import matplotlib.pyplot as plt
import seaborn as sns

# Nombre de variables
n = len(variables)

# Calculer le nombre de lignes et de colonnes pour la disposition des sous-figures
cols = 4  # Nombre de colonnes
rows = (n // cols) + (n % cols)  # Nombre de lignes nécessaires

# Boxplots
plt.figure(figsize=(16, 5 * rows))
for i, var in enumerate(variables, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x="AGE_GROUPE_2", y=var, data=df_cleaned)
    plt.ylabel(var)
    plt.title(f'Boxplot de {var}')  # Titre de chaque graphique

plt.tight_layout()  # Ajuster l'espacement entre les graphiques
plt.show()

# Violinplots
plt.figure(figsize=(16, 5 * rows))
for i, var in enumerate(variables, 1):
    plt.subplot(rows, cols, i)
    sns.violinplot(x="AGE_GROUPE_2", y=var, data=df_cleaned, inner="quartile")
    plt.ylabel(var)
    plt.title(f'Violinplot de {var}')  # Titre de chaque graphique

plt.tight_layout()  # Ajuster l'espacement entre les graphiques
plt.show()

#######_____________________Question 3.2_______________________


import pandas as pd

# Créer une fonction pour attribuer un groupe d'âge
def age_group(age):
    if 6 <= age < 8 :
        return '6-7 ans'
    elif 8 <= age < 10:
        return '8-9 ans'
    elif 10 <= age < 12:
        return '10-11 ans'
    return 'Autre'

df_cleaned['AGE_GROUPE_3'] = df_cleaned['Age'].apply(age_group)


print(df_cleaned["AGE_GROUPE_3"].value_counts())


import pandas as pd

# Statistiques descriptives pour chaque groupe d'âge
desc_stats = df_cleaned.groupby('AGE_GROUPE_3')[variables].describe()

# Afficher les statistiques descriptives
print(desc_stats)


from scipy import stats

# Liste pour enregistrer les résultats
results2 = []

# Nous appliquons  le test de Kruskal-Wallis ou ANOVA pour chaque variable d'anticipation

for var in variables:
    if df_cleaned[var].dtype in ['float64', 'int64']:
        # Vérifier la normalité avec le test de Shapiro
        stat, p_value = stats.shapiro(df_cleaned[var])  # Test de normalité
        if p_value > 0.05:
            # Normalité acceptée : appliquer l'ANOVA
            test_result = stats.f_oneway(df_cleaned[df_cleaned['AGE_GROUPE_3'] == '6-7 ans'][var],
                                         df_cleaned[df_cleaned['AGE_GROUPE_3'] == '8-9 ans'][var],
                                         df_cleaned[df_cleaned['AGE_GROUPE_3'] == '10-11 ans'][var])
            test_stat = test_result.statistic
            test_p_value = test_result.pvalue
            test_type = 'ANOVA'
        else:
            # Non-normalité : appliquer le test de Kruskal-Wallis
            test_result = stats.kruskal(df_cleaned[df_cleaned['AGE_GROUPE_3'] == '6-7 ans'][var],
                                         df_cleaned[df_cleaned['AGE_GROUPE_3'] == '8-9 ans'][var],
                                         df_cleaned[df_cleaned['AGE_GROUPE_3'] == '10-11 ans'][var])
            test_stat = test_result.statistic
            test_p_value = test_result.pvalue
            test_type = 'Kruskal-Wallis'
        
        # Ajouter les résultats dans la liste
        results2.append({
            'Variable': var,
            'Test': test_type,
            'Statistic': test_stat,
            'p-value': test_p_value
        })

# Convertir les résultats en DataFrame pour les afficher sous forme de tableau
test_results_df = pd.DataFrame(results2)



import matplotlib.pyplot as plt
import seaborn as sns

# Boxplot pour comparer les groupes d'âge pour la variable Taju

# Nombre de variables
n = len(variables)
# Calculer le nombre de lignes et de colonnes pour la disposition des sous-figures
cols = 4  # Nombre de colonnes
rows = (n // cols) + (n % cols)  # Nombre de lignes nécessaires
# Créer la figure avec haute résolution
plt.figure(figsize=(16, 12), dpi=200)

for i, var in enumerate(variables, 1):
    plt.subplot(rows, cols, i)
    sns.boxplot(x="AGE_GROUPE_3", y=var, data=df_cleaned)
    plt.ylabel(var)
    plt.title(f'Boxplot de {var}')  # Titre de chaque graphique
plt.tight_layout()  # Ajuster l'espacement entre les graphiques
plt.show()

# Nombre de variables
n = len(variables)
# Calculer le nombre de lignes et de colonnes pour la disposition des sous-figures
cols = 4  # Nombre de colonnes
rows = (n // cols) + (n % cols)  # Nombre de lignes nécessaires
# Créer la figure avec haute résolution
plt.figure(figsize=(16, 12), dpi=200)
for i, var in enumerate(variables, 1):
    plt.subplot(rows, cols, i)
    sns.violinplot(x="AGE_GROUPE_3", y=var, data=df_cleaned, inner="quartile")
    plt.ylabel(var)
    plt.title(f'Violinplot de {var}')  # Titre de chaque graphique
plt.tight_layout()  # Ajuster l'espacement entre les graphiques
plt.show()


#########______________________Question 3.3___________________

import statsmodels.api as sm
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Initialiser un DataFrame pour stocker les résultats
regression_results = pd.DataFrame(columns=['Variable', 'Coef_Age', 'P-value_Age', 'R-squared'])

# Définir le nombre de lignes et de colonnes pour l'affichage des graphiques
rows, cols = 5, 4  
n = len(variables)

# Créer la figure avec plusieurs sous-graphes
fig, axes = plt.subplots(rows, cols, figsize=(20, 25))  
axes = axes.flatten()  # Aplatir la matrice pour parcourir plus facilement les sous-graphes

# Boucle pour effectuer les régressions et afficher les graphiques
for i, var in enumerate(variables):
    # Ajouter une colonne d'ordonnée à l'origine (intercept) uniquement pour X
    X = df_cleaned[['Age']].copy()
    X = sm.add_constant(X)  # Ajoute une colonne 'const' pour l'intercept
    
    y = df_cleaned[var]

    # Ajuster le modèle
    model = sm.OLS(y, X).fit()

    # Ajouter les résultats dans le tableau
    new_row = pd.DataFrame([{
        'Variable': var,
        'Coef_Age': round(model.params['Age'], 3),
        'P-value_Age': round(model.pvalues['Age'], 3),
        'R-squared': round(model.rsquared, 3),
    }])
    regression_results = pd.concat([regression_results, new_row], ignore_index=True)

    # Tracer la régression sur le sous-graphe correspondant
    ax = axes[i]
    sns.scatterplot(x=df_cleaned['Age'], y=y, ax=ax, alpha=0.5, label='Données')
    sns.lineplot(x=df_cleaned['Age'], y=model.fittedvalues, ax=ax, color='red', label='Régression')

    ax.set_title(f"Régression de {var} en fonction de l'âge")
    ax.set_xlabel('Âge')
    ax.set_ylabel(var)

# Ajustement automatique des sous-graphes
plt.tight_layout()
plt.show()

########________________Question 3.4______________________

import statsmodels.api as sm
import seaborn as sns
import matplotlib.pyplot as plt

# --- Étape 1 : Régression linéaire simple ---

variables_taille_poids = ['Height', 'Weight']
Results = []

for vari in variables_taille_poids:
    X = df_cleaned[['Age']]
    X = sm.add_constant(X)  # Ajoute l'intercept
    y = df_cleaned[var]
    
    model = sm.OLS(y, X).fit()  # Ajustement du modèle
    
    # Stocker les résultats
    Results.append({
        'Variable': var,
        'Coef_Age': round(model.params['Age'], 3),
        'P-value_Age': round(model.pvalues['Age'], 3),
        'R-squared': round(model.rsquared, 3)
    })
    
    # Affichage du graphique
    plt.figure(figsize=(6, 4))
    sns.scatterplot(x=df_cleaned['Age'], y=y, alpha=0.5, label='Données')
    sns.lineplot(x=df_cleaned['Age'], y=model.fittedvalues, color='red', label='Régression')
    plt.xlabel('Âge')
    plt.ylabel(vari)
    plt.title(f"Régression de {vari} en fonction de l'âge")
    plt.legend()
    plt.show()

# Afficher les résultats sous forme de tableau
import pandas as pd
regression_results_simple = pd.DataFrame(Results)
print(regression_results_simple)

# --- Étape 2 : Régression multiple ---
X_taille = df_cleaned[['Age', 'Weight']]
X_poids = df_cleaned[['Age', 'Height']]

X_taille = sm.add_constant(X_taille)
X_poids = sm.add_constant(X_poids)

model_taille = sm.OLS(df_cleaned['Height'], X_taille).fit()
model_poids = sm.OLS(df_cleaned['Weight'], X_poids).fit()

# Affichage des résultats de la régression multiple
print("\nRégression Taille ~ Age + Poids\n", model_taille.summary())
print("\nRégression Poids ~ Age + Taille\n", model_poids.summary())
