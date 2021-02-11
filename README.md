# Modelling and Solving the Vehicle Routing Problem (VRP)

Project for the "Decision Making of Constraint Programming" course, University of Bologna (2020/2021)

## Assignment

Modeling and solving the vehicle routing problem in MiniZinc (using the MiniZinc IDE or the Python API) with Gecode.

* Given:
  * a setof customers, each with a:
    * demand (e.g.,the quantity required);
    * location;
  * a fleet of vehicles, each with a capacity,
    determine:
  * the vehicle serving to each customer;
  * the route of each vehicle,
* so as to minimize the total cost:
  * e.g., total distance, total distance + number of vehicles.
* assuming that each vehicle departs from and returns to the depot.

## Given Data

* n customers: N = {1,.., n}
* n customer demands: D = {d1,.., n}
* m vehicles (and routes): V = {1,.., m}
* m vehicle capacities: C = {c1,…, cm}
* n+1 locations: L = {l1, l2,…, ln, ln+1}
  * Each customer i is represented as a pair of x and y coordinates.
  * The final location is that of the depot.

### We Also Need

* Distance between every Ii and Ij.
  * Calculate using the coordinates.
  * Multiply by 1000 and round it.
* Total visits = n+2m
* Z = {1,.., n, n+1,.., n+m, n+m+1,.., n+2m}

## Developers

* Cotugno Giosuè
* [Pruscini Davide](https://github.com/prushh/)

## Setup

To execute the script, Python must be installed([download](https://www.python.org/downloads/)), and some external libraries must be downloaded and installed using the pip (or pip3) package manager:

```bash
pip install -r requirements.txt
```

It is also need to have installed MiniZinc ([download](https://www.minizinc.org/)): it is important to have MiniZinc usable from teminal.
To do that you have to add the path of the MiniZinc installation folder to the PATH environment variable ([here](https://www.minizinc.org/doc-2.4.3/en/installation.html)).

## Usage

```bash
python preprocessing.py dzn_path [-o OUTPUT] [[-s] -m MODEL] [-l LIMIT] [-or {customers, demands, distances}] [-p]
```

Below are some examples, to learn more about all the avaible parameters refer to the help.

```bash
python preprocessing.py -h
```

### Examples

#### 1. Preprocessing only, the default output file is named tmp.dzn

```bash
python preprocessing.py prXX.dzn
```

#### 2. Preprocessing and searching for solutions with the specified model, a time limit of 300s is set

```bash
python preprocessing.py prXX.dzn -sm vrp.mzn -l 300
```

#### 3. Preprocessing and searching for solutions with the specified model, setting a time limit of 300s, sorting the dataset by nearest customer and finally plot the routes for each vehicle

```bash
python preprocessing.py prXX.dzn -sm vrp.mzn -l 300 -or customers -p
```
