# Journaling

## séance 1:
 - installation des environnements de dév

## séance 2:

### Objectifs:
- [x] Faire un noeud de téléop (joy -> simulation)
- [x] Faire un launchfile
    - téléop
    - joy
    - map
    - (simu)
- [x] Récupérer occupancy grid

## séance 3:
### Objectifs:
 - [x] Créer un graphe à partir de la carte (fonction voisinnage)
 - [ ] Implémentation de A*

### Notes:
A* c'est pas prometteur, début d'implém de rrt* pendant les vacances par Maxime
Nxgraph c'est bien mais c'est pas clair

## séance 4: (2024)
 - [X] Fin de l'implémentation de RRT_Star
 - [X] Optimisation de RRT_Star
 - [ ] Refactor de la méthode de rewire (opt)
 - [ ] Refactor de la fonction collision (opt, voir BIT_star)

### Notes:
On parcours en avant et en arrière le chemin retourné par l'algo de path finding pour déterminer le chemin le plus court et éviter de passer par des points inutiles.

## séance 5:
### Objectifs: 
 - [ ] Préprocessing de la carte (ouverture et touti quenti)
 - [ ] Transformation des coordonés des pts retournés par le path finding dans le repère monde (transfo carte -> odom -> baselink)
 - [ ] Commencer le suivi de chemin 🫰
