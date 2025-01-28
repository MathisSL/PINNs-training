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
    Les données d'entrée incluent des mesures partielles des vitesses (u,vu,v) dans un champ donné.
    Le réseau est entraîné en minimisant une fonction de perte qui combine l'erreur par rapport aux données et le respect des contraintes physiques (les résidus des équations).

The code implements a PINN to approximate the solution to the Navier-Stokes equations. It combines data-driven learning (from velocity and pressure data) with physical constraints imposed by the Navier-Stokes equations:

## Navier-Stokes Equations

- Momentum equations:

- Continuity equation:

The PINN predicts the velocity  and pressure  fields while enforcing these equations as soft constraints in the loss function.

## Code Structure

### Libraries

The code imports the following:

torch: For building and training the neural network.

numpy: For numerical data handling.

scipy.io: For loading MATLAB .mat data files.

matplotlib: For visualization and animation.

### Physical Constant
```pytyhon
nu = 0.01 : Kinematic viscosity, a parameter used in the Navier-Stokes equations.
```
### NavierStokes

The class implements the PINN for solving the Navier-Stokes equations.

Initialization
```python
def __init__(self, X, Y, T, u, v):
```
Inputs:

X, Y, T: Spatial and temporal coordinates of training points.

u, v: Velocity components at the training points.

Converts inputs into PyTorch tensors and initializes the neural network and optimizer.

Network Architecture
```python
def network(self):
    self.net = nn.Sequential(
        nn.Linear(3, 20), nn.Tanh(),
        ...,
        nn.Linear(20, 2))
```
A fully connected neural network with:

Input: .

Hidden layers: 9 layers, each with 20 neurons and Tanh activation.

Output: , where  is the stream function, and  is pressure.

Physics-Informed Function
```python
def function(self, x, y, t):
```
Calculates the following:



Velocities:



Partial derivatives (e.g., ) using torch.autograd.grad.

Residuals:

: Residual of the momentum equation in the -direction.

: Residual of the momentum equation in the -direction.

Loss Function
```pyhton
def closure(self):
```
Computes the loss as:

Backpropagates the loss to update network weights.

Training
```python
def train(self):
```
Uses the **LBFGS** optimizer to minimize the loss function and train the networ that is **one of the most used optimizer for PINNs**.

3. Data Preparation

data = scipy.io.loadmat('cylinder_wake.mat')
X_star, U_star, P_star, t_star = data['X_star'], data['U_star'], data['p_star'], data['t']

Loads training data from a .mat file, which contains:

X_star: Spatial coordinates.

U_star: Velocity components.

P_star: Pressure values.

t_star: Time values.

Training data is sampled randomly:

idx = np.random.choice(N * T, N_train, replace=False)
x_train, y_train, t_train = x[idx], y[idx], t[idx]
u_train, v_train = u[idx], v[idx]

4. Training and Inference

Training

pinn = NavierStokes(x_train, y_train, t_train, u_train, v_train)
pinn.train()
torch.save(pinn.net.state_dict(), 'model.pt')

Trains the PINN and saves the trained model to model.pt.

Inference

pinn.net.load_state_dict(torch.load('model.pt'))
pinn.net.eval()
u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, t_test)

Loads the trained model and performs predictions for test data.

5. Visualization and Animation

Static Plot

plt.contourf(u_plot, levels=30, cmap='jet')
plt.colorbar()
plt.show()

Visualizes the pressure field  as a static contour plot.

Animation

def animate(i):
    u_out, v_out, p_out, f_out, g_out = pinn.function(x_test, y_test, i * t_test)
    ...
ani = animation.FuncAnimation(fig, animate, 20, interval=1, blit=False)
plt.show()

Animates the evolution of the pressure field over time.

6. Summary

This code demonstrates the use of a PINN to solve the Navier-Stokes equations. It combines:

A neural network to approximate velocity and pressure fields.

Physics-based constraints to ensure the predictions satisfy the Navier-Stokes equations.

Data-driven learning for increased accuracy.

The result is a model capable of predicting fluid dynamics in complex systems.
