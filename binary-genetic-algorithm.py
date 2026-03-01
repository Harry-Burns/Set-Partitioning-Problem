import numpy as np
from src.setup import load_sppwn


(N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL = (None,None),None,None,None

INDIVIDUAL_SHAPE =  None; POPULATION_SIZE = None; 
NUM_PARENTS = None; NUM_CHILDREN = None; NUM_ELITES = None
POPULATION_SHAPE = None; INIT_SELECT_RATE = None


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

        #if penalty[i] < penalty[j]: winner = i
        #elif penalty[j] < penalty[i]: winner = j
        #else: winner = i if cost[i] < cost[j] else j
        winner = i if cost[i] + penalty[i] * 2000 < cost[j] + penalty[j] * 2000 else j

        parents.append(p[winner])
    return np.array(parents)

def better_mutation(p, pm, r_min, r_max):
    N_POP = p.shape[0]

    for i in range(N_POP):
        num_rows = np.random.randint(r_min,r_max+1)
        rows = np.random.randint(N_ROWS, size=num_rows)

        for r in rows:
            cover = np.where(COL_ROWS[:, r])[0]
            selected_cover = cover[p[i, cover]]

            s = selected_cover.size

            if s == 0:
                c = np.random.choice(cover)
                p[i,c] = True

            elif s == 1:
                c_off = selected_cover[0]
                p[i,c_off] = False
        
                cover_diff = cover[cover != c_off]
                c_on = np.random.choice(cover_diff) if cover_diff.size else c_off
                p[i,c_on] = True

            else:
                c = np.random.choice(selected_cover)
                p[i,c] = False
    return p

def mutation(p, pm): # based on   "Lecture5_EvolutionaryAlgorithms-6.pdf" Slide 20/25, using fixed pm
    bit_flips = np.random.rand(*p.shape) < pm
    p[bit_flips] = ~p[bit_flips]
    return p

def better_crossover(parents, num_children): # based on   "Lecture5_EvolutionaryAlgorithms-6.pdf" Slide 21/25, using uniform crossover
    children = []

    def parent_crossover(x1, x2):
        # Crossovers push feasibility
        x1_sums = x1.astype(int) @ COL_ROWS
        x2_sums = x2.astype(int) @ COL_ROWS

        xc = np.zeros_like(x1, dtype=int)
        t = np.arange(N_ROWS)

        while(t.size):
            r = np.random.choice(t)
            x1r = x1_sums[r]; x2r = x2_sums[r]
            if x1r > x2r:
                candidates = np.where(x1 & COL_ROWS[:, r])[0]
            elif x2r > x1r:
                candidates = np.where(x2 & COL_ROWS[:, r])[0]
            else:
                parent = x1 if np.random.rand() > 0.5 else x2
                candidates = np.where(parent & COL_ROWS[:,r])[0]

            covered = (xc @ COL_ROWS) > 0
            safe = candidates[~np.any(COL_ROWS[candidates][:,covered], axis=1)]
            candidates = safe if safe.size else candidates

            if candidates.size:
                c = np.random.choice(candidates)
                xc[c] = True; t = t[~COL_ROWS[c,t]]
            
            t = t[t != r]

        return xc.astype(bool)

    while len(children) < num_children:
        i, j = np.random.choice(len(parents), 2, replace=False)
        c1 = parent_crossover(parents[i], parents[j])

        children.append(c1)

    return np.array(children)

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

def replacement(p, p_cost, p_pen, p_child, p_child_cost, p_child_pen):
    elite_idx = np.lexsort((p_cost, p_pen))[:NUM_ELITES]
    remaining = POPULATION_SIZE - NUM_ELITES

    # Merge current population (non-elites) with children
    non_elite_idx = np.lexsort((p_cost, p_pen))[NUM_ELITES:]
    combined = np.vstack([p[non_elite_idx], p_child])
    combined_cost = np.concatenate([p_cost[non_elite_idx], p_child_cost])
    combined_pen = np.concatenate([p_pen[non_elite_idx], p_child_pen])

    rank_idx = np.lexsort((combined_cost, combined_pen))
    selected_idx = rank_idx[:remaining]

    p = np.vstack([p[elite_idx], combined[selected_idx]])
    p_cost = np.concatenate([p_cost[elite_idx], combined_cost[selected_idx]])
    p_pen = np.concatenate([p_pen[elite_idx], combined_pen[selected_idx]])

    return p, p_cost, p_pen

def elite_replacement(p, p_cost, p_pen, p_child, p_child_cost, p_child_pen):
    # --- Replacement (in p_fit and p) using elitism.
    elite_idx = np.lexsort((p_cost,p_pen))[:NUM_ELITES]
    remaining = POPULATION_SIZE - NUM_ELITES

    rank_idx = np.lexsort((p_child_cost,p_child_pen))
    selected_idx = rank_idx[:remaining]

    p = np.vstack([p[elite_idx], p_child[selected_idx]])
    p_cost = np.concatenate([p_cost[elite_idx], p_child_cost[selected_idx]])
    p_pen = np.concatenate([p_pen[elite_idx], p_child_pen[selected_idx]])

    return p, p_cost, p_pen
    # ----- 

def ranking_replacement(p, p_cost, p_pen, p_child, p_child_cost, p_child_pen):
    def pick_worst(mask):
        idx = np.flatnonzero(mask)
        order = np.lexsort((p_cost[idx], p_pen[idx]))
        return idx[order[-1]]
    
    for i,(cost,pen) in enumerate(zip(p_child_cost,p_child_pen)):
        g1 = (p_cost >= cost) &( p_pen >= pen)
        g2 = (p_cost < cost )& (p_pen >= pen )
        g3 = (p_cost >= cost) &( p_pen < pen )
        g4 = (p_cost < cost )& (p_pen < pen  )
    
        if any(g1):
            j = pick_worst(g1)
        elif any(g2):
            j = pick_worst(g2)
        elif any(g3):
            j = pick_worst(g3)
        elif any(g4):
            j = pick_worst(g4)

        p[j] = p_child[i]
        p_cost[j] = cost
        p_pen[j] = pen
    return p, p_cost, p_pen
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
    diversity = float('inf')

    while gen < max_iter:
        p_parent = selection(p, p_cost, p_pen, NUM_PARENTS)

        p_new = better_crossover(p_parent, NUM_CHILDREN)
        p_new = better_mutation(p_new, 0.5, 1, 10)

        p_new_cost, p_new_pen = fitness(p_new, zero_w=zero_w, overlap_w=overlap_w)
        p, p_cost, p_pen = replacement(p, p_cost, p_pen, p_new, p_new_cost, p_new_pen)

        # --- Tracking best
        best_i = np.lexsort((p_cost, p_pen))[0]
        best = p[best_i]; cost_best = p_cost[best_i]; pen_best = p_pen[best_i]

        # --- Logging
        if gen % 25 == 0 and verbose:
            diversity = diversity_hamming_mean(p) # it doesnt matter if we do an extra few generations
            feas_percent = np.where(p_pen == 0)[0].size/len(p_pen)
            avg_cost = np.mean(p_cost)
            print(f"Cost: {cost_best:.0f} | Penalty: {pen_best:.0f} | {'X' if pen_best != 0 else 'Y'} | Gen: {gen} | Feasible %: {feas_percent:.3f} | Diversity: {diversity:.2f} | Avg Cost: {avg_cost:.0f}")#Penalty Weights: {zero_w:.0f},{overlap_w:.0f} ")
        
        gen += 1

        if pen_best == 0 and cost_best <= TARGET_OPTIMAL:
            return best, gen
        
        # --- Restart population when diversity gets too low, but keep elites.
        if diversity < 1:
            print(f"Diversity too low \"{gen}\" - exiting...")
            return best,gen
            #p_elite = p[np.lexsort((p_cost, p_pen))[:NUM_ELITES]]
            #p = np.random.rand(*POPULATION_SHAPE) < INIT_SELECT_RATE
            #p[:NUM_ELITES] = p_elite
            #p_cost, p_pen = fitness(p, zero_w=zero_w, overlap_w=overlap_w)
            #diversity = float('inf')  # reset
            #continue

    return best, gen
# --------------------


import csv

for f_name in ['sppnw42', 'sppnw41', 'sppnw43']:
    print(f"Beginning Processing {f_name}...")
    
    output_path = f'results/bga_{f_name}_results.csv'

    (N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL = load_sppwn(f_name)

    # --- Genetic Algorithm Parameters
    INDIVIDUAL_SHAPE = (N_COLS)

    POPULATION_SIZE = 512

    NUM_PARENTS = POPULATION_SIZE // 4
    NUM_CHILDREN = POPULATION_SIZE * 4

    NUM_ELITES = POPULATION_SIZE // 128

    POPULATION_SHAPE = (POPULATION_SIZE, INDIVIDUAL_SHAPE)

    INIT_SELECT_RATE = 10.0 / N_COLS # only want to start with around 10 columns selected, otherwise we get stuck in massively costly and unfit solutions
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