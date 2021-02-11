include "globals.mzn";
include "gecode.mzn";

string: Name;
array [int] of float: locX; % locX[n+1] = depotX
array [int] of float: locY; % locY[n+1] = depotY
array [int] of int: Demand; % Customer demands
int: NumVehicles; % Vehicle (and routes)
array [1..NumVehicles] of int: Capacity; % Vehicle capacities
int: MaxDistance;

int: N=length(locX)-1; % Number of Customers

set of int: CUSTOMER = 1..N;
set of int: VEHICLE = 1..NumVehicles;
set of int: VEHICLE_SYM = 1..NumVehicles-1;
set of int: LOAD = 0..max(Capacity);

set of int: NODES = 1..N+2*NumVehicles;
set of int: START_NODES = N+1..N+NumVehicles;
set of int: START_NODES_PREPATH = N+2..N+NumVehicles;
set of int: END_NODES = N+NumVehicles+1..N+2*NumVehicles;
set of int: END_NODES_PATH = N+NumVehicles+1..N+2*NumVehicles-1;

% Distances between every I[i] and I[j]
array [1..N+1, 1..N+1] of int: Distances;
array [NODES, NODES] of int: distance = array2d(NODES, NODES,[
    if i <= N /\ j <= N then
        Distances[i+1, j+1]
    elseif i == N+1 /\ j == N+1 then
        Distances[1,1]
    elseif i <= N /\ j >= N+1 then
        Distances[i+1, 1]
    elseif j <= N /\ i >= N+1 then
        Distances[1, j+1]
    else
        Distances[1, 1]
    endif
    | i,j in NODES
]);

array [NODES] of var NODES: path;
array [NODES] of var NODES: prePath;
array [NODES] of var VEHICLE: vehicleRoute;
array [NODES] of var LOAD: demandEachTruck;
var float: obj;

%---------------------------------------------------------------------------
% GLOBAL CONSTRAINTS
% Create a big Hamiltonian Cycle between all nodes
% in the graph, including the dummy nodes.
constraint circuit(path);
% Use prePath for indicate that every start nodes come before an end nodes.
constraint circuit(prePath);

% PATH and PREPATH CONSTRAINTS, TSP-Problem
% Each end node has as successor a start node, excluding the last one.
constraint forall(i in (END_NODES_PATH))(
    path[i] = i-NumVehicles+1
);

% Last end node in path is equal to the first start node.
constraint path[N+2*NumVehicles] = N+1;

% Consistence between successor and predecessor.
constraint redundant_constraint(
    forall(i in NODES)(
        path[prePath[i]] = i
    ) /\
    forall(i in NODES)(
        prePath[path[i]] = i
    )
);

% Each predecessor in start nodes, excluding the first one,
% is equal to an end node.
constraint forall(i in (START_NODES_PREPATH))(
    prePath[i] = i + NumVehicles - 1
);

% First start node has as predecessor is equal the last end node.
constraint prePath[N+1] = N+2*NumVehicles;

%---------------------------------------------------------------------------
% VEHICLES CONSTRAINTS
% Each vehicle has one start node.
constraint forall(i in START_NODES)(
    vehicleRoute[i] = i-N
);

% Each vehicle has one end node.
constraint forall(i in END_NODES)(
    vehicleRoute[i] = i-N-NumVehicles
);

% Next vehicle is define by the previous one.
constraint forall(i in CUSTOMER)(
    vehicleRoute[path[i]] = vehicleRoute[i]
);

% Previous vehicle is define by the successor one.
constraint redundant_constraint(
    forall(i in CUSTOMER)(
        vehicleRoute[prePath[i]] = vehicleRoute[i]
    )
);

%---------------------------------------------------------------------------
% CAPACITY CONSTRAINTS - Knapsack
% Each partial load of vehicle is 0 when it's
% in the depot.
constraint forall(i in START_NODES)(
    demandEachTruck[i] = 0
);

% Useful to initilize the first visited node
% by a vehicle to 0.
constraint forall(i in START_NODES)(
    demandEachTruck[i] = demandEachTruck[path[i]]
);

% Update the partial load of vehicle with the sum
% of the previous load and the old demand.
constraint forall(i in CUSTOMER)(
    demandEachTruck[i] + Demand[i] = demandEachTruck[path[i]]
);

% Check that partial load is always less or equal
% then the capacity for each vehicle.
constraint redundant_constraint(
    forall(i in CUSTOMER)(
        demandEachTruck[i] <= Capacity[vehicleRoute[i]]
    )
);

% Check that final load is less or equal
% then the capacity for each vehicle.
constraint forall(i in VEHICLE)(
    demandEachTruck[i+N+NumVehicles] <= Capacity[i]
);

%---------------------------------------------------------------------------
% SYMMETRY BREAKING CONSTRAINT
% Usage Vehicle: a vehicle is available only if
% the previous one is used.
constraint symmetry_breaking_constraint(
    forall(i in VEHICLE_SYM)(
        path[N+i] = N+NumVehicles+i -> path[N+(i+1)] = N+NumVehicles+(i+1)
    )
);

%---------------------------------------------------------------------------
% OBJECTIVE FUNCTION
% Find how much vehicles are in use.
var int: usedVehicles;
constraint usedVehicles = max([vehicleRoute[j] | j in CUSTOMER]);

% Sum the distance of the path that the solver found.
var int: dist;
constraint dist = sum([distance[i, path[i]] |i in NODES]);

% Linear scalarization, multi-objective optimization
var float: ALPHA = 0.5;
var float: BETA = 0.5;
constraint obj = ((ALPHA * (dist/MaxDistance)) + (BETA * (usedVehicles/NumVehicles)));

%---------------------------------------------------------------------------
% HEURISTIC SEARCH, RESTART and MINIMIZE
solve   ::  seq_search([
                int_search(path, first_fail, indomain_split),
                int_search(vehicleRoute, first_fail, indomain_split),
                int_search(demandEachTruck, first_fail, indomain_random),
                relax_and_reconstruct(path, 94),
                restart_luby(N*4)
            ])
        minimize obj;

output [
    "PATH:" ++ show(path) ++ "\n" ++
    "PRE_PATH:" ++ show(prePath) ++ "\n" ++
    "VEHICLE:" ++ show(vehicleRoute) ++ "\n" ++
    "DEMAND:" ++ show(Demand) ++ "\n" ++
    "DEMAND_EACH_TRUCK:" ++ show(demandEachTruck) ++ "\n" ++
    "MAX_DIST:" ++ show(MaxDistance) ++ "\n" ++
    "USED_VEHICLES:" ++ show(usedVehicles) ++ "\n" ++
    "DIST:" ++ show(dist) ++ "\n" ++
    "OBJ:" ++ show(obj) ++ "\n"
];