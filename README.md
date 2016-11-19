# Coding Dojo - Réseaux de neurones


### Installation

Ce dojo requière une installation de `python2` et de `pip`. La version 2 de python est utilisée puisque les datasets ont été
compressés avec `python2` et ne sont pas compatibles avec `python3`.

```
$ sudo pip2 install virtualenv
$ virtualenv -p python2 .env
```

Les commandes précédentes créent un environnement virtuel pour isoler l'installation des dépendances.
Il suffit ensuite d'activer l'environnement virtuel et d'installer les dépendances:

```
$ source .env/bin/activate
$ pip install -r requirements.txt
```


### Liens

Ce dojo est basé en partie sur le matériel du cours de Stanford CS231n. http://cs231n.stanford.edu/syllabus.html
