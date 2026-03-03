import os
import csv
import numpy as np

def load_sppwn(f_name):
    file_path = f'data/{f_name}.txt'

    try:
        with open(file_path, 'r') as f:
            data = f.readlines()
    except FileNotFoundError as fnfe:
        print(f"File not found in '/data/{f_name}'', trying '/{f_name}.txt'...")
        file_path = f'{f_name}.txt'
        try:
            with open(file_path, 'r') as f:
                data = f.readlines()
        except FileNotFoundError as fnfe:
            print("File not found!")
            raise FileNotFoundError("Data file not found. Exiting...")

    data = [[int(x) for x in d.strip().split()] for d in data]
    
    N_ROWS,N_COLS = data[0]

    COL_COSTS = np.array([d[0] for d in data[1:]])
    COL_ROWS = np.array([np.isin(np.arange(1,N_ROWS+1), d[2:]) for d in data[1:]])

    if f_name == 'sppnw41': TARGET_OPTIMAL = 11307
    elif f_name == 'sppnw42': TARGET_OPTIMAL = 7656
    elif f_name == 'sppnw43': TARGET_OPTIMAL = 8904
    else: TARGET_OPTIMAL = 0

    return (N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL


(N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL = (None,None),None,None,None

INDIVIDUAL_SHAPE =  None; POPULATION_SIZE = None; 
NUM_PARENTS = None; NUM_CHILDREN = None; NUM_ELITES = None
POPULATION_SHAPE = None; MUTATION_RATE = None; INIT_SELECT_RATE = None
STOCHASTIC_ITER = None; STOCHASTIC_PROB = None


# --- Help Handle Termination (Quit if diversity is too low)
def diversity_hamming_mean(p: np.ndarray) -> float:
    pop, _ = p.shape
    ones = p.sum(axis=0)
    zeros = pop - ones
    return float((2 * ones * zeros).sum() / (pop * (pop - 1)))
# --------


# ---- Requirement 3.1 - Initialization Algorithm
def init_individual() -> np.ndarray:
    individual = np.zeros(INDIVIDUAL_SHAPE, dtype=bool)
    rows = np.arange(N_ROWS)

    while rows.size != 0:
        row_sums = np.sum(COL_ROWS[individual], axis=0)
        r = np.random.choice(rows)

        candidates = np.where(COL_ROWS[:, r])[0]
        safe = candidates[np.all((COL_ROWS[candidates] == 0) | (row_sums == 0), axis=1)]

        if safe.size > 0:
            col = np.random.choice(safe)
            individual[col] = True
            cov_rows = np.where(COL_ROWS[col])[0]
            rows = np.setdiff1d(rows, cov_rows)
        else:
            rows = np.setdiff1d(rows, [r])

    return individual

def initialization(pop_size):
    return np.array([init_individual() for i in range(pop_size)], dtype=bool)
# ----

# ---- Requirement 3.2 - Stochastic Ranking
def stochastic_ranking(costs: np.ndarray, penalties: np.ndarray) -> np.ndarray:
    I = np.random.permutation(costs.shape[0])

    for i in range(STOCHASTIC_ITER):
        has_swapped = False

        for j in range(costs.shape[0] - 1):
            a,b = I[j], I[j+1]

            u = np.random.rand()

            if (penalties[a] == 0 and penalties[b] == 0) or u < STOCHASTIC_PROB:
                if costs[a] > costs[b]:
                    I[j], I[j+1] = I[j+1], I[j]
                    has_swapped = True
            else:
                if penalties[a] > penalties[b]:
                    I[j], I[j+1] = I[j+1], I[j]
                    has_swapped = True

        if not has_swapped:
            break

    return I
# ----

# ---- Requirement 3.3 - Heuristic Improvement Operator
COL_COST_PER_ROW = None
def heuristic_improvement_operator(solution: np.ndarray) -> np.ndarray:
    w = np.sum(COL_ROWS[solution], axis=0)

    # Drop overlapping rows
    t = np.where(solution)[0]
    while t.size:
        col = np.random.choice(t)
        t = t[t != col]

        if np.any(w[COL_ROWS[col]] >= 2):
            solution[col] = False
            w[COL_ROWS[col]] -= 1

    u = np.where(w == 0)[0]
    v = u.copy()

    while v.size > 0:
        r = np.random.choice(v)
        v = v[v != r]

        candidates = np.where(COL_ROWS[:, r])[0]

        not_in_u = np.setdiff1d(np.arange(N_ROWS), u)
        safe_costs = [(c,COL_COST_PER_ROW[c]) for c in candidates if np.all(~COL_ROWS[c, not_in_u])] # All the rows not_in_u must be 0.

        best = min(safe_costs, key=lambda x: x[1])[0] if safe_costs else None

        if best is not None:
            solution[best] = True
            rows = np.where(COL_ROWS[best,:])[0]
            w[rows] += 1
            u = np.setdiff1d(u, rows)
            v = np.setdiff1d(v, rows)
    
    return solution
def heuristic(p: np.ndarray) -> np.ndarray:
    return np.array([heuristic_improvement_operator(x.copy()) for x in p])
# ----

# ---- Fitness
def cost(p: np.ndarray) -> np.ndarray:
    return p.astype(int) @ COL_COSTS

def fitness(p: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    row_sums = p.astype(int) @ COL_ROWS
    costs = p.astype(int) @ COL_COSTS

    zero_pen = (row_sums < 1).sum(axis=-1)
    overlap_pen = np.maximum(row_sums - 1, 0).sum(axis=-1)

    penalties = zero_pen + overlap_pen

    return costs,penalties
# ----

# --- Genetic Algorithm Functions
def selection(p, pos, num_parents):
    N = p.shape[0]
    parents = []

    for _ in range(num_parents):
        i, j = np.random.randint(0, N, 2)
        winner = i if pos[i] < pos[j] else j
        parents.append(p[winner])

    return np.array(parents)

def mutation(p, pm): # based on   "Lecture5_EvolutionaryAlgorithms-6.pdf" Slide 20/25, using fixed pm
    bit_flips = np.random.rand(*p.shape) < pm
    p[bit_flips] = ~p[bit_flips]
    return p

def crossover(parents, num_children): # based on   "Lecture5_EvolutionaryAlgorithms-6.pdf" Slide 21/25, using uniform crossover
    children = []

    def parent_crossover(x1, x2):
        mask = np.random.rand(INDIVIDUAL_SHAPE) < 0.5
        c1 = x1.copy()
        c2 = x2.copy()
        c1[mask] = x2[mask]
        c2[mask] = x1[mask]
        return c1, c2

    while len(children) < num_children:
        i, j = np.random.choice(len(parents), 2, replace=False)
        c1, c2 = parent_crossover(parents[i], parents[j])

        children.append(c1)

        if len(children) < num_children:
            children.append(c2)

    return np.array(children)
# -------------

# Improved BGA Algorithm
def improved_binary_genetic_algorithm(p0: np.ndarray, max_iter=10000, verbose=False):
    if p0.shape != POPULATION_SHAPE:
        raise ValueError("Initialisation population shape is invalid.") 
    
    p = p0; gen = 0

    p_cost,p_pen = fitness(p)
    p_rank = stochastic_ranking(p_cost,p_pen)

    while gen < max_iter:
        p_pos = np.argsort(p_rank)

        p_parent = selection(p, p_pos, NUM_PARENTS)
        p_new = crossover(p_parent, NUM_CHILDREN)
        p_new = mutation(p_new, pm=MUTATION_RATE) 
        p_new = heuristic(p_new)

        p_combined = np.vstack([p,p_new])

        p_combined_cost,p_combined_pen = fitness(p_combined)
        p_combined_rank = stochastic_ranking(p_combined_cost,p_combined_pen)

        elite_idx = elite_idx = np.lexsort((p_cost, p_pen))[:NUM_ELITES]
        remaining = POPULATION_SIZE - NUM_ELITES

        rem_rank_idx = p_combined_rank[~np.isin(p_combined_rank, elite_idx)]
        selected_idx = rem_rank_idx[:remaining]

        p = np.vstack([p[elite_idx], p_combined[selected_idx]])

        p_cost,p_pen = fitness(p)
        p_rank = stochastic_ranking(p_cost,p_pen)

        # Tracking best & logging
        best_i = np.lexsort((p_cost, p_pen))[0]
        best = p[best_i]; cost_best = p_cost[best_i]; pen_best = p_pen[best_i]

        diversity = diversity_hamming_mean(p)
        if gen % 10 == 0 and verbose:
            feas_percent = np.where(p_pen == 0)[0].size/len(p_pen)
            avg_cost = np.mean(p_cost)
            print(f"Cost: {cost_best:.0f} | Penalty: {pen_best:.0f} | {'X' if pen_best != 0 else 'Y'} | Gen: {gen} | Feasible %: {feas_percent:.3f} | Diversity: {diversity:.2f} | Avg Cost: {avg_cost:.0f}")
        #Penalty Weights: {zero_w:.0f},{overlap_w:.0f} ")
        # ----
        
        gen += 1

        if pen_best == 0 and cost_best <= TARGET_OPTIMAL:
            return best, gen
        
        if diversity < 4:
            print("Minimum diversity reached. Exiting...")
            return best, gen

    return best, gen
# --------------------



for f_name in ['sppnw42', 'sppnw43']:
    print(f"Beginning Processing {f_name}...")
    
    output_path = f'results/ibga_{f_name}_results.csv'

    (N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL = load_sppwn(f_name)

    # --- Genetic Algorithm Parameters
    INDIVIDUAL_SHAPE = (N_COLS)

    POPULATION_SIZE = 128

    NUM_PARENTS = POPULATION_SIZE // 4
    NUM_CHILDREN = 256
    NUM_ELITES = POPULATION_SIZE // 100

    POPULATION_SHAPE = (POPULATION_SIZE, INDIVIDUAL_SHAPE)

    MUTATION_RATE = 4.0 / N_COLS #N_COLS # want around 2-6 columns changed per mutation
    MAX_ITER = 1000

    STOCHASTIC_PROB = 0.45
    STOCHASTIC_ITER = POPULATION_SIZE // 2
    # --------------

    # --- Heuristic Parameters
    coverage = np.sum(COL_ROWS, axis=1)
    COL_COST_PER_ROW = np.divide(COL_COSTS, coverage, where=coverage > 0)
    COL_COST_PER_ROW[coverage == 0] = np.inf
    # --------

    # NUMBER OF TRIALS
    TRIALS = 30

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'cost', 'gens', 'feasible', 'num_columns', 'overlaps', 'uncovered', 'columns_used'])

    for x in range(TRIALS):
        print(f"Current Trial: {x}")

        p0 = initialization(POPULATION_SIZE)
        best, num_gens = improved_binary_genetic_algorithm(p0=p0, max_iter=MAX_ITER, verbose=True)

        best_cols = COL_ROWS[best]
        row_sums = np.sum(best_cols, axis=0)

        feasible = np.all(row_sums == 1)
        cols_used = np.where(best)[0]


        with open(output_path, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([
                x,
                cost(best),
                num_gens,
                feasible,
                cols_used.size,
                (row_sums > 1).sum(),
                (row_sums < 1).sum(),
                ' '.join(map(str, cols_used))   # store BEST as column indices
            ])
