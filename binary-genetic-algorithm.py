import numpy as np
from src.setup import load_sppwn


(N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL = (None,None),None,None,None

INDIVIDUAL_SHAPE =  None; POPULATION_SIZE = None; 
NUM_PARENTS = None; NUM_CHILDREN = None; NUM_ELITES = None
POPULATION_SHAPE = None; MUTATION_RATE = None; INIT_SELECT_RATE = None


# Fitness Functions
def cost(p: np.ndarray) -> np.ndarray:
    return p.astype(int) @ COL_COSTS

def fitness(p: np.ndarray, zero_w: int=1, overlap_w: int=1) -> tuple[np.ndarray, np.ndarray]: # Function now acts on a population at a time
    row_sums = p.astype(int) @ COL_ROWS
    costs = cost(p)
    zero_pen = (row_sums < 1).sum(axis=1)
    overlap_pen = np.maximum(row_sums - 1, 0).sum(axis=1)
    penalty = zero_w * zero_pen + overlap_w * overlap_pen
    return costs, penalty
# --------------------




# --- Genetic Algorithm Functions
def selection(p, cost, penalty, num_parents):
    N = p.shape[0]
    parents = []
    for _ in range(num_parents):
        i, j = np.random.randint(0, N, 2)

        if penalty[i] == penalty[j]:
            winner = i if cost[i] < cost[j] else j
        elif penalty[i] < penalty[j] and cost[i] < cost[j]:
            winner = i
        elif penalty[j] < penalty[i] and cost[j] < cost[i]:
            winner = j
        else:
            winner = i # equivalent to 50/50

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
        c1 = x1.copy(); c2 = x2.copy()
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
# --------------------


# --- Help Handle Termination (Quit if diversity is too low)
def diversity_hamming_mean(p: np.ndarray) -> float:
    pop, _ = p.shape
    ones = p.sum(axis=0)
    zeros = pop - ones
    return float((2 * ones * zeros).sum() / (pop * (pop - 1)))
# --------


# --- Binary Genetic Algorithm
def binary_genetic_algorithm(p0: np.ndarray|None=None, max_iter: int=10000, verbose=False):
    zero_w,overlap_w = 1,1 #4000, 3000

    p = p0 if p0 is not None else np.random.rand(*POPULATION_SHAPE) < INIT_SELECT_RATE
    p_cost, p_pen = fitness(p, zero_w=zero_w, overlap_w=overlap_w)

    gen = 0

    while gen < max_iter:
        p_parent = selection(p, p_cost, p_pen, NUM_PARENTS)
        p_new = crossover(p_parent, NUM_CHILDREN)
        p_new = mutation(p_new, pm=MUTATION_RATE) 

        p_new_cost, p_new_pen = fitness(p_new, zero_w=zero_w, overlap_w=overlap_w)

        # Replacement (in p_fit and p) using elitism.
        p_combined = np.vstack([p,p_new])
        p_cost_combined = np.concatenate([p_cost,p_new_cost])
        p_pen_combined = np.concatenate([p_pen,p_new_pen])

        rank_idx = np.lexsort((p_cost_combined,p_pen_combined)) # Best are feasible, next best have good fitness
        elite_idx = np.lexsort((p_cost,p_pen))[:NUM_ELITES]

        remaining = POPULATION_SIZE - NUM_ELITES
        rank_idx = rank_idx[~np.isin(rank_idx,elite_idx)] # works because p is before p_new
        selected_idx = rank_idx[:remaining]

        p = np.vstack([p[elite_idx], p_combined[selected_idx]])
        p_cost = np.concatenate([p_cost[elite_idx], p_cost_combined[selected_idx]])
        p_pen = np.concatenate([p_pen[elite_idx], p_pen_combined[selected_idx]])
        # ----


        # --- Tracking best
        p_fit = p_cost + 1000 * p_pen
        
        feas_idx = np.where(p_pen == 0)[0]
        if feas_idx.size > 0:   best_i = feas_idx[np.argmin(p_cost[feas_idx])]
        else:                   best_i = np.argmin(p_fit)
        best = p[best_i]; cost_best = p_cost[best_i]; pen_best = p_pen[best_i]; fit_best = p_fit[best_i]

        # --- Logging
        diversity = diversity_hamming_mean(p)
        if gen % 25 == 0 and verbose:
            feas_percent = np.where(p_pen == 0)[0].size/len(p_pen)
            pop_child_proportion = (len(selected_idx[selected_idx >= len(p_cost)]) / (len(selected_idx) + len(elite_idx)))
            avg_cost = np.mean(p_cost)
            print(f"Cost: {cost_best:.0f} | Penalty: {pen_best:.0f} | Fit: {fit_best:.0f} | {'X' if cost_best != fit_best else 'Y'} | Gen: {gen} | Feasible %: {feas_percent:.3f} | Diversity: {diversity:.2f} | Pop Child Proportion: {pop_child_proportion:.3f} | Avg Cost: {avg_cost:.0f}")#Penalty Weights: {zero_w:.0f},{overlap_w:.0f} ")
        
        gen += 1

        if fit_best == cost_best and cost_best <= TARGET_OPTIMAL:
            return best, gen
        
        if diversity < 1:
            print("Minimum diversity reached. Exiting...")
            return best, gen

    return best, gen
# --------------------


import csv

for f_name in ['sppnw42', 'sppnw41', 'sppnw43']:
    print(f"Beginning Processing {f_name}...")
    
    output_path = f'results/bga_{f_name}_results.csv'

    (N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL = load_sppwn(f_name)

    # --- Genetic Algorithm Parameters
    INDIVIDUAL_SHAPE = (N_COLS)

    POPULATION_SIZE = 2048

    NUM_PARENTS = 256
    NUM_CHILDREN = 2048
    NUM_ELITES = 4

    POPULATION_SHAPE = (POPULATION_SIZE, INDIVIDUAL_SHAPE)

    MUTATION_RATE = 2.0 / N_COLS #4.0 / N_COLS #N_COLS # want around 2-6 columns changed per mutation
    INIT_SELECT_RATE = 20.0 / N_COLS # only want to start with around 10 columns selected, otherwise we get stuck in massively costly and unfit solutions
    # ---

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'cost', 'gens', 'feasible', 'num_columns', 'overlaps', 'uncovered', 'columns_used'])


    TRIALS = 30

    for x in range(TRIALS):
        print(f"Current Trial: {x}")

        best, num_gens = binary_genetic_algorithm(max_iter=2000, verbose=True)

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