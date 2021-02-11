# -*- coding: utf-8 -*-
"""
CVRP - preprocessing and execution with MiniZinc and Gecode solver
Authors: Cotugno Giosuè, Pruscini Davide
"""

import math
import argparse
from os import path
from ast import literal_eval

import pymzn
import matplotlib.pyplot as plt

from utils import Point, on_segment, orientation, do_intersect


DZN_EXT = '.dzn'
MZN_EXT = '.mzn'
MIN_LIMIT = 10
MAX_LIMIT = 600


def range_limited_time(arg: str) -> int:
    '''
    Type function for argparse - a int within some predefined bounds.
    '''
    try:
        s = int(arg)
    except ValueError:    
        raise argparse.ArgumentTypeError("must be a int point number")
    if s < MIN_LIMIT or s > MAX_LIMIT:
        raise argparse.ArgumentTypeError(f"argument must be < {str(MAX_LIMIT)} and > {str(MIN_LIMIT)} (seconds)")
    return s


def file_exists(file_path: str, wanted_ext: str) -> bool:
    '''
    Check if file exists and it's a \'.dzn\' file.
    '''
    if path.exists(file_path):
        _, ext = path.splitext(file_path)
        if ext == wanted_ext:
            return True
        raise Exception(f"{file_path} file is not a {wanted_ext}")
    raise Exception(f"{file_path} doesn't exists in {path.abspath(file_path)[0:-len(file_path)]}")


def load_dataset(dzn_path: str) -> dict:
    '''
    Read the .dzn file and save it in a dict.
    (pymzn.dzn2dict doesn't support 'string' type for the conversion).
    '''
    if file_exists(dzn_path, DZN_EXT):
            tmp, data = [], {}

            with open(dzn_path, 'r') as in_file:
                for i,elm in enumerate(in_file):
                    # Ignore the 'Name' entry
                    if i != 0:
                        tmp.append(elm)

            for i,elm in enumerate(tmp):
                if i == 0:
                    data['locX'] = list(literal_eval(elm[7:-2]))
                elif i == 1:
                    data['locY'] = list(literal_eval(elm[7:-2]))
                elif i == 2:
                    data['Demand'] = list(literal_eval(elm[9:-2]))
                elif i == 3:
                    data['NumVehicles'] = int(elm[14:-2])
                elif i == 4:
                    # Some files have a comma after the last element
                    # of the Capacity array, we remove it
                    data['Capacity'] = list(literal_eval(elm[11:-1].replace(",]", "]")))
            return data

def _get_distances(loc_x: list, loc_y: list) -> list:
    '''
    Return a matrix with the distances between each node,
    using euclidean distance formula.
    Depot is in the first column.
    '''
    N = len(loc_x)
    matrix = [[0 for i in range(N)] for j in range(N)]
    for i in range(N):
        for j in range(N):
            matrix[i][j] = round(math.sqrt((loc_x[i]-loc_x[j])**2 + (loc_y[i]-loc_y[j])**2)*1000)
    return matrix


def _get_pseudo_path(dist: list, N: int, check_cost: bool = False) -> list:
    '''
    Useful for calculate distance of approximate longest route,
    returns a list containing the indices of the designated route (sorting by customers).
    '''

    taken = [True if i == 0 else False for i in range(N)]
    new_index = [0 for i in range(N)]

    for i in range(N):
        if i != 0:
            index_min = 0
            row = 0 if check_cost else 9999999
            for j in range(N):
                if check_cost:
                    # Condition for hightes distance
                    if (dist[new_index[i-1]][j] >=  row and
                        not taken[j]):
                        index_min = j
                        row = dist[new_index[i-1]][j]
                else:
                    # Condition for nearest customer
                    if (dist[new_index[i-1]][j] < row and
                        not taken[j]):
                        index_min = j
                        row = dist[new_index[i-1]][j]
            taken[index_min] = True
            new_index[i] = index_min
    
    return new_index

def _reassign_callback(old: list, index: list, N: int, flag: bool = False) -> list:
    '''
    Reassign calculate value by index list, used for locX, locY and demand.
    '''
    new = []
    for i in range(N):
        if not flag:
            new.append(old[index[i]])
        else:
            new.append(old[index[i+1]-1])
    return new


def process(ds: dict, dzn_name: str, order: str) -> dict:
    '''
    Organises the data read from the .dzn file according
    to the specified order parameter.
    It also adds some new variabile for the MiniZinc model.
    '''
    
    N = len(ds['locX'])
    distances = _get_distances(ds['locX'], ds['locY'])
    
    # Get route with highest distance
    route_cost = 0
    new_index = _get_pseudo_path(distances[:], N, check_cost=True)
    
    for i in range(N-1):
            route_cost += distances[new_index[i]][new_index[i+1]]
    route_cost += distances[new_index[N-1]][new_index[0]]

    if order == 'distances':
        # Ordering by decreasing distances
        index = [x for x in range(N-1)]
        distances_from_depot = [0 for x in range(N-1)]

        for i in range(N-1):
            distances_from_depot[i] = distances[N-1][i]

        tmp = distances_from_depot
        tmp = sorted(tmp, reverse=True)

        for i in range(N-1):
            for j in range(N-1):
                if distances_from_depot[i] == tmp[j]:
                    index[i] = j

        index_dep = index[:]
        index_dep.insert(0, N-1)

        ds['locX'] = _reassign_callback(ds['locX'][:], index_dep, N)
        ds['locY'] = _reassign_callback(ds['locY'][:], index_dep, N)
        ds['Demand'] = _reassign_callback(ds['Demand'][:], index, N-1)

    elif order == 'customers':
        # Ordering by nearest customer (better)
        ds['locX'].insert(0, ds['locX'].pop())
        ds['locY'].insert(0, ds['locY'].pop())
        distances = _get_distances(ds['locX'], ds['locY'])

        new_index = _get_pseudo_path(distances[:], N)

        ds['locX'] = _reassign_callback(ds['locX'][:], new_index, N)
        ds['locY'] = _reassign_callback(ds['locY'][:], new_index, N)
        ds['Demand'] = _reassign_callback(ds['Demand'][:], new_index, N-1, flag=True)

    elif order == 'cusdenoc':
        # Ordering by customers, demands and no crossing
        ds['locX'].insert(0, ds['locX'].pop())
        ds['locY'].insert(0, ds['locY'].pop())
        points = []

        distances = _get_distances(ds['locX'], ds['locY'])

        for i in range(N):
            tmp= Point(ds['locX'][i],ds['locY'][i])
            points.append(tmp)

        ds['Capacity'] = sorted(ds['Capacity'], reverse=True)
        dem=ds['Demand'][:]
        capacity=ds['Capacity'][:]
        NumVehicles=len(capacity)

        z=0
        act_cap=[0 for i in range(200)]
        for i in range(100):
            capacity.append(200)

        taken = [True if i == 0 else False for i in range(N)]
        new_index = [0 for i in range(N)]
        matrix_crossing=[[0 for i in range(N)]for j in range(N)]
        
        for i in range(N):
            if i != 0:
                index_min = 0
                row_min = 9999999
                crossing_min = 999999
                for j in range(N):
                    for k in range(j-1):
                        if do_intersect(points[new_index[i-1]],points[j],points[k],points[k+1]):
                            matrix_crossing[new_index[i-1]][j]+=1

                    if(z>= NumVehicles or act_cap[z]+dem[j-1]<=capacity[z]):
                        if 0==j:
                            if distances[new_index[i-1]][j] <  row_min and not taken[j]:
                                index_min = j
                                row_min = distances[new_index[i-1]][j]
                        else:
                            if ((matrix_crossing[new_index[i-1]][j] < crossing_min or
                                distances[new_index[i-1]][j] < row_min) and
                                not taken[j]):
                                index_min = j
                                row_min = distances[new_index[i-1]][j]
                                crossing_min = matrix_crossing[new_index[i-1]][j]
                    else:
                        z+=1
                        for j in range(N):
                            if ( distances[0][j] <  row_min) and not taken[j]:
                                index_min = j
                                row_min = distances[0][j]
                                crossing_min=matrix_crossing[new_index[i-1]][j]
                taken[index_min] = True
                new_index[i] = index_min

        ds['locX'] = _reassign_callback(ds['locX'][:], new_index, N)
        ds['locY'] = _reassign_callback(ds['locY'][:], new_index, N)
        ds['Demand'] = _reassign_callback(ds['Demand'][:], new_index, N-1, flag=True)

    elif order == 'demands':
        # Ordering by decreasing demands
        dict_demand = {}

        for i in range(N-1):
            dict_demand[i] = ds['Demand'][i]

        dict_demand = dict(sorted(dict_demand.items(), key=lambda item: item[1], reverse=True))
        index = list(dict_demand.keys())

        demand = ds['Demand'][:]
        for i in range(N-1):
            ds['Demand'][i] = demand[index[i]]

        index_dep = index[:]
        index_dep.insert(0, N-1)

        ds['locX'] = _reassign_callback(ds['locX'][:], index_dep, N)
        ds['locY'] = _reassign_callback(ds['locY'][:], index_dep, N)
    else:
        # No sorting, put the depot in first position
        ds['locX'].insert(0, ds['locX'].pop())
        ds['locY'].insert(0, ds['locY'].pop())
        distances = _get_distances(ds['locX'], ds['locY'])

    ds['Name'] = "\"" + dzn_name + "\""
    ds['Distances'] = _get_distances(ds['locX'], ds['locY'])
    ds['Capacity'] = sorted(ds['Capacity'], reverse=True)
    ds['MaxDistance'] = route_cost
    
    return ds


def plot_path(ds: dict, path: list, vehicle: list):
    '''
    Plot for each vehicle the path found as solution.
    '''
    def _dark_subplots(nrows: int = 1, ncols: int = 1) -> tuple:
        '''
        Create subplots and set dark theme.
        '''
        plt.style.use('dark_background')
        fig, axes = plt.subplots()
        fig.patch.set_facecolor('#252526')
        axes.set_facecolor('#3c3c3c')

        return (fig, axes)

    N = len(ds['locX'])
    LAST_ROUTE = len(path)-1
    NODES = len(path[0])
    VEHICLE = ds['NumVehicles']

    # Initialize index with key for each vehicle
    index = {}
    for j in range(VEHICLE):
        index[j+1] = [0 for i in range(NODES)]
    
    # Get route for each vehicle
    for j in range(VEHICLE):
        for i in range(NODES):
            if vehicle[LAST_ROUTE][i] == j+1:
                index[j+1][i] = path[LAST_ROUTE][i]

    # Remove for each vehicle the end_node from the path
    for j in range(VEHICLE):
        for i in range(VEHICLE):
            index[i+1].pop()
    
    # Get only first two nodes
    for j in range(VEHICLE):
        for i in range(N, N+VEHICLE):
            if index[j+1][i-1] != 0:
                index[j+1].insert(0, index[j+1][i-1])
                index[j+1].insert(0, i)
                index[j+1][i+1] = 0
                break

    # Get start_node and first route point, and then
    # get the rest of the path for each vehicle
    first_two = []
    for j in range(VEHICLE):
        for i in range(NODES-1):
            if i < 2:
                first_two.append(index[j+1][i])
            elif first_two[i-1]+1 == first_two[0]+VEHICLE+1:
                index[j+1] = first_two[:]
                first_two = []
                break
            else:
                first_two.append(index[j+1][first_two[i-1]+1])

    # Get coordinates of route points for each vehicle
    loc_x, loc_y = {}, {}
    for j in range(VEHICLE):
        LEN_ROUTE = len(index[j+1])
        loc_x[j], loc_y[j] = [], []
        for i in range(LEN_ROUTE):
                if index[j+1][i] < N:
                    loc_x[j].append(ds['locX'][index[j+1][i]])
                    loc_y[j].append(ds['locY'][index[j+1][i]])
                else:
                    loc_x[j].append(ds['locX'][0])
                    loc_y[j].append(ds['locY'][0])

    # Settings for plot
    plot_title = f"Routes for {ds['Name'][1:-1]}"
    window_title = ds['Name'][1:-1]
    fig, ax = _dark_subplots()
    fig.suptitle(plot_title, fontsize=15)
    fig.canvas.set_window_title(window_title)

    # Print annotation with arrow, one color for each vehicle
    cmap = plt.cm.get_cmap('tab20c', VEHICLE+1)
    for j in range(VEHICLE):
        ax.plot(loc_x[j], loc_y[j],
                color=cmap(j),
                marker='o',
                linewidth=.1,
                markersize=5)
        N = len(loc_x[j])-1
        for i in range(N):
            ax.annotate('',
                        xy=(loc_x[j][i+1], loc_y[j][i+1]),
                        xytext=(loc_x[j][i], loc_y[j][i]),
                        arrowprops=dict(
                            arrowstyle="->",
                            color=cmap(j),
                            lw=1.75
                        )
            )    

    plt.show()


def print_solution(sols: pymzn.Solutions) -> (list, list):
    '''
    If MiniZinc find solutions, print them with also
    the statistics (failures, num_solutions, ecc.).
    Return two lists for plotting purpose.
    '''
    path, vehicle = [], []
    for i in range(len(sols)):
        path.append(sols[i]['path'])
        vehicle.append(sols[i]['vehicleRoute'])
    sols.print(log=True)

    return path, vehicle


def main() -> int:
    '''
    Sequential operation:
        - load .dzn file
        - process it and save it in a new file
        - try to find solutions with MiniZinc and Gecode solver (optional)
        - plot in a graph all path obtained for visual feedback (optional)
    '''
    data = load_dataset(args.dzn_path)

    if not data:
        raise Exception(f"an error occurred while loading the dataset")

    data = process(data, args.dzn_path, args.order)
    # Save processed data to {args.output}.dzn file
    pymzn.dict2dzn(data, fout=args.output)
    
    print("Preprocessing completed.")
    if not args.solve:
        return -1
    
    if not file_exists(args.model, MZN_EXT):
        return -1

    print(f"Running {args.model} with {args.output}({args.dzn_path})...")
    # Default solver is Gecode (pymzn.Solver)
    sols = pymzn.minizinc(args.model, args.output, timeout=args.limit, all_solutions=True)

    if not sols:
        print("No solutions found.")
        return -1
    
    path, vehicle = print_solution(sols)
    if args.plot:
        plot_path(data, path, vehicle)

    return 0


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Dataset preprocessing and MiniZinc execution.',
        usage='%(prog)s dzn_path [-o OUTPUT] [[-s] -m MODEL] [-l LIMIT] [-or {distances, demands, customers, cusdenoc}] [-p]'
    )
    parser.add_argument(
        'dzn_path', type=str,
        help='the file containing the dataset'
    )
    parser.add_argument(
        '-o','--output', type=str,
        default='tmp.dzn',
        help='the processed file containing the dataset'
    )
    parser.add_argument(
        '-s','--solve', default=False,
        action='store_true',
        help='try to find solutions with MiniZinc'
    )
    parser.add_argument(
        '-m', '--model', type=str,
        help='the file containing the model'
    )
    parser.add_argument(
        '-l','--limit', default=60,
        type=range_limited_time,
        help='execution timelimit in seconds (default: 60s)'
    )
    parser.add_argument(
        '-or', '--order', default='none', type=str,
        choices=['distances', 'demands', 'customers', 'cusdenoc'],
        help='order dataset for decreasing distances, nearest customer or decreasing demands (default: none)'
    )
    parser.add_argument(
        '-p','--plot', default=False,
        action='store_true',
        help='plot graph with route of each vehicle'
    )

    args = parser.parse_args()
    if args.solve and (args.model is None):
        parser.error("specify the file containing the model")

    if args.model and (not args.solve):
        parser.error("specify -s for execute the model with MiniZinc")
    
    exit(main())
