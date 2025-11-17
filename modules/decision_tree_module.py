import pandas as pd
import numpy as np
from math import log2
from sklearn.preprocessing import LabelEncoder
from sklearn import tree
import matplotlib
matplotlib.use("Agg")   # Use non-GUI backend for web apps
import matplotlib.pyplot as plt
import base64
import io
from flask import Flask, jsonify

# =========================
# DATASET PLAY GOLF
# =========================
data = {
    'Outlook': ['Sunny','Sunny','Overcast','Rainy','Rainy','Rainy','Overcast','Sunny',
                'Sunny','Rainy','Sunny','Overcast','Overcast','Rainy'],
    'Temperature': ['Hot','Hot','Hot','Mild','Cool','Cool','Cool','Mild',
                    'Cool','Mild','Mild','Mild','Hot','Mild'],
    'Humidity': ['High','High','High','High','Normal','Normal','Normal','High',
                 'Normal','Normal','Normal','High','Normal','High'],
    'Windy': ['False','False','True','False','False','True','True','False',
              'False','False','True','True','False','True'],
    'PlayGolf': ['No','No','Yes','Yes','Yes','No','Yes','No','Yes',
                 'Yes','Yes','Yes','Yes','No']
}

df = pd.DataFrame(data)

# =========================
# ENTROPY
# =========================
def entropy(target):
    elements, counts = np.unique(target, return_counts=True)
    ent = 0
    for i in range(len(elements)):
        p = counts[i] / np.sum(counts)
        ent -= p * log2(p)
    return round(ent, 4)

# =========================
# INFORMATION GAIN
# =========================
def info_gain(df, feature, target="PlayGolf"):
    total_entropy = entropy(df[target])
    vals, counts = np.unique(df[feature], return_counts=True)

    weighted_entropy = 0
    for i in range(len(vals)):
        subset = df[df[feature] == vals[i]]
        weighted_entropy += (counts[i] / np.sum(counts)) * entropy(subset[target])

    return round(total_entropy - weighted_entropy, 4)

# =========================
# ID3 ALGORITHM
# =========================
def ID3(df, features, target="PlayGolf"):
    # Jika semua label sama
    if len(df[target].unique()) == 1:
        return df[target].iloc[0]

    # Jika fitur habis
    if len(features) == 0:
        return df[target].mode()[0]

    # Cari feature terbaik
    gains = [info_gain(df, f, target) for f in features]
    best_feature = features[gains.index(max(gains))]

    tree_struct = {best_feature: {}}
    for value in df[best_feature].unique():
        subset = df[df[best_feature] == value]
        new_features = [f for f in features if f != best_feature]
        subtree = ID3(subset, new_features, target)
        tree_struct[best_feature][value] = subtree

    return tree_struct

# =========================
# SKLEARN VISUALIZATION
# =========================
def sklearn_tree_image():
    le = LabelEncoder()
    df_encoded = df.apply(le.fit_transform)

    X = df_encoded.drop(columns=['PlayGolf'])
    y = df_encoded['PlayGolf']

    clf = tree.DecisionTreeClassifier(criterion="entropy")
    clf.fit(X, y)

    fig, ax = plt.subplots(figsize=(10, 6))
    tree.plot_tree(clf, feature_names=list(X.columns),
                   class_names=["No", "Yes"],
                   filled=True)

    buffer = io.BytesIO()
    fig.savefig(buffer, format='png')   # save from fig, not plt
    buffer.seek(0)

    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    plt.close(fig)

    return img_base64

# =========================
# API RESPONSE
# =========================
def get_decision_tree_data():
    features = list(df.columns[:-1])
    manual_tree = ID3(df, features)

    gains = {f: info_gain(df, f) for f in features}

    sklearn_img = sklearn_tree_image()

    return {
        "features": features,
        "gains": gains,
        "manual_tree": manual_tree,
        "sklearn_tree": sklearn_img
    }

