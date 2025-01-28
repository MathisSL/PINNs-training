# PINNs-training
Les PINNs (Physics-Informed Neural Networks) sont des réseaux de neurones qui combinent les mathématiques de la physique avec l'intelligence artificielle pour résoudre des problèmes complexes.

En gros : ce sont des modèles d'IA qui respectent les lois physiques, comme les équations de la mécanique des fluides, la chaleur ou l'élasticité.

## Comment ça marche ?

Équations physiques : On commence avec une équation qui décrit un phénomène physique (par exemple, l'équation de la chaleur ou de Navier-Stokes pour les fluides).
Réseau de neurones : On entraîne un réseau de neurones à prédire des solutions possibles (températures, vitesses, pressions, etc.).
Contraintes physiques : On guide le réseau en lui disant : "Tes prédictions doivent respecter l'équation physique." Cela sert de "professeur" pour corriger le réseau.

## Avantages

  Moins de données : Pas besoin de tonnes de mesures réelles, car les lois physiques remplacent une partie des données.
  Applications diverses : On peut les utiliser pour modéliser des phénomènes comme les flux d'air autour d'une voiture, les mouvements d'une vague, ou la propagation de la chaleur.



![PINNS_1](https://github.com/user-attachments/assets/81bbcdac-71d0-4d03-9e06-163e5ba97cda)

![PINNS_2](https://github.com/user-attachments/assets/fd964920-6b15-4258-b62b-baf7fa5710b9)

-----
# Résoudre les équations de Navier Strocks avec PINNs
## Équations mises en jeu

Les équations de Navier-Stokes résolues par le PINN (Physics-Informed Neural Network) sont :
Conservation de la quantité de mouvement (en x et y)

$\frac{\partial u}{\partial t} + u \frac{\partial u}{\partial x} + v \frac{\partial u}{\partial y} = -\frac{\partial p}{\partial x} + \nu \left( \frac{\partial^2 u}{\partial x^2} + \frac{\partial^2 u}{\partial y^2} \right)$
    

Équation en y :

$\frac{\partial v}{\partial t} + u \frac{\partial v}{\partial x} + v \frac{\partial v}{\partial y} = -\frac{\partial p}{\partial y} + \nu \left( \frac{\partial^2 v}{\partial x^2} + \frac{\partial^2 v}{\partial y^2} \right)$


Incompressibilité (condition de divergence nulle) :}


$\frac{\partial u}{\partial x} + \frac{\partial v}{\partial y} = 0$



Problème résolu

![p_field_lbfgs](https://github.com/user-attachments/assets/957f3595-775d-4437-9200-8ea7bff323b1)


Le code résout un problème d'écoulement fluide bidimensionnel autour d'un obstacle (un cylindre dans ce cas) en utilisant un réseau de neurones informé par la physique. L'objectif est d'estimer les champs de vitesse u(x,y,t), v(x,y,t) et de pression p(x,y,t) à partir de données d'observation partielles tout en respectant les lois physiques.
En résumé :

  Les prédictions du modèle (u, v, p) doivent satisfaire les équations de Navier-Stokes.
    Les données d'entrée incluent des mesures partielles des vitesses (u,v) dans un champ donné.
    Le réseau est entraîné en minimisant une fonction de perte qui combine l'erreur par rapport aux données et le respect des contraintes physiques (les résidus des équations).

# Explications du code
## 1. Initialisation et chargement des données

Les données d'entraînement proviennent d'un fichier .mat contenant les champs de vitesse et de pression générés autour d'un cylindre. Ces données sont extraites et réorganisées pour servir de base d'entraînement.

```python
data = scipy.io.loadmat('cylinder_wake.mat')
U_star = data['U_star']  # Champ de vitesse (N x 2 x T)
P_star = data['p_star']  # Champ de pression (N x T)
t_star = data['t']       # Temps (T x 1)
X_star = data['X_star']  # Coordonnées (N x 2)
```
Les données sont ensuite sélectionnées aléatoirement pour former un ensemble d'entraînement :
```python
idx = np.random.choice(N * T, N_train, replace=False)
x_train = x[idx, :]
y_train = y[idx, :]
t_train = t[idx, :]
u_train = u[idx, :]
v_train = v[idx, :]
```
## 2. Classe NavierStokes

La classe NavierStokes est la structure principale pour entraîner un réseau de neurones afin de résoudre les équations de Navier-Stokes. Voici ses composants :
### a. Initialisation

Les données d'entraînement (x, y, t, u, v) sont converties en tenseurs PyTorch et déplacées vers le GPU ou CPU, selon la disponibilité.
```python
self.x = torch.tensor(X, dtype=torch.float32, requires_grad=True).to(device)
self.u = torch.tensor(u, dtype=torch.float32).to(device)
```
### b. Réseau de neurones

Le réseau est défini comme une architecture entièrement connectée avec 10 couches et la fonction d'activation Tanh. Chaque entrée est une concaténation des coordonnées spatiales et temporelles (x, y, t), et la sortie est composée de deux composantes : le potentiel de vitesse (psi) et la pression (p).
```python
self.net = nn.Sequential(
    nn.Linear(3, 20), nn.Tanh(),
    nn.Linear(20, 20), nn.Tanh(),
    ...
    nn.Linear(20, 2)
).to(self.device)
```
### c. Fonction physique : Résolution des équations

Les dérivées nécessaires aux équations de Navier-Stokes sont calculées via torch.autograd.grad. Cela permet de respecter les contraintes physiques :
```python
f = u_t + u * u_x + v * u_y + p_x - nu * (u_xx + u_yy)
g = v_t + u * v_x + v * v_y + p_y - nu * (v_xx + v_yy)
```
f et g représentent les équations de Navier-Stokes à respecter, où :

  u,v sont les composantes de vitesse.
  p est la pression.
  ν est la viscosité cinématique.

### d. Optimisation

Le réseau est entraîné en minimisant une fonction de perte basée sur l'erreur quadratique moyenne (MSE). Les termes de perte incluent :

  Les erreurs entre les prédictions des vitesses et des valeurs réelles.
  Les erreurs entre les équations physiques (f, g) et des vecteurs nuls.
```python
u_loss = self.mse(u_prediction, self.u)
v_loss = self.mse(v_prediction, self.v)
f_loss = self.mse(f_prediction, self.null)
g_loss = self.mse(g_prediction, self.null)
```
Un optimiseur **LBFGS** est utilisé pour entraîner le modèle, c'est l'**un des plus utilisé en PINNs**.
## 3. Entraînement et prédiction
Entraînement

Pour entraîner le modèle :
```python
pinn.train()
torch.save(pinn.net.state_dict(), 'model.pt')
```
Chargement d'un modèle pré-entraîné

un modèle a déjà été entraîné, il peut être chargé pour faire des prédictions :
```python
pinn.net.load_state_dict(torch.load('model.pt'))
```
## 4. Visualisation des résultats

Les prédictions de pression sont visualisées sous forme de contours colorés. L'animation montre l'évolution temporelle.
Exemple de tracé statique :
```python
plt.contourf(u_plot, levels=30, cmap='jet')
plt.colorbar()
plt.show()
```
Animation :

Une animation est générée avec matplotlib.animation.FuncAnimation :
```python
ani = animation.FuncAnimation(fig, animate, 20, interval=1, blit=False)
plt.show()
```
Utilisation

  Placer le fichier de données cylinder_wake.mat dans le même dossier que le script.
  Décommenter la ligne pour entraîner le modèle si nécessaire :
```python
  pinn.train()
```
  Exécuter le script pour visualiser les résultats et l'animation.

Résultats

Le modèle PINN produit des champs de pression et de vitesse en respectant les équations de Navier-Stokes. L'animation montre comment ces champs évoluent au fil du temps.
