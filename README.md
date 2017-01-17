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
---
### Installation avec Docker

Une alternative à l'installation proposée ci-haut est l'utilisation d'un *container* [Cloud Datalab](https://github.com/googledatalab/datalab) 
local sur `Docker`. Le Datalab est un environnement pour le Machine Learning en Python créé par Google.
#### Mac OSX

##### Installez Docker et Kitematic
[Kitematic](https://kitematic.com/) est une interface utilisateur pour Docker permettant d'éviter d'utiliser le docker-cli.
###### Avec [Homebrew](http://brew.sh/)
```
$ brew cask install docker
$ brew cask install kitematic
```
Si un des package est introuvable, pensez à faire `brew update`<br/>

###### Sans Homebrew
Téléchargez et installez le docker toolbox [ici](https://www.docker.com/products/docker-toolbox)

##### Installez le container Cloud Datalab
```
$ docker run -it -p "127.0.0.1:8081:8080" -v "${HOME}:/content" \
  -e "PROJECT_ID=<coding-dojo>" \
  gcr.io/cloud-datalab/datalab:local
```
Le Datalab roule maintenant sur *localhost:8081*. Vous pouvez également utiliser Kitematic pour
le démarrer. Vous avez maintenant un environnement comprennant toutes les librairies nécessaires
pour ce dojo sans même avoir installé Python.

---
### Liens

Ce dojo est basé en partie sur le matériel du cours de Stanford CS231n. http://cs231n.stanford.edu/syllabus.html
