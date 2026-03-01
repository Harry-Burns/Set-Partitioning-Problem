import numpy as np
from src.setup import load_sppwn

(N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL = (None,None),None,None,None

# Fitness Functions
def cost(solution: np.ndarray) -> int:
    return np.sum(COL_COSTS[solution])

def penalty(solution: np.ndarray, zero_w: int=1, overlap_w: int=1) -> int:
    cols = COL_ROWS[solution]
    row_sums = np.sum(cols, axis=0)

    zero_rows = row_sums[row_sums < 1]
    overlap_rows = row_sums[row_sums > 1]

    return zero_rows.size * zero_w + np.sum(overlap_rows - 1) * overlap_w

def fitness(solution: np.ndarray, zero_w: int, overlap_w: int) -> int:
    return cost(solution) + penalty(solution,zero_w=zero_w,overlap_w=overlap_w)
# --------------------


# Neighbour Function
def neighbour(solution: np.ndarray, k=4) -> np.ndarray: # Using inspiration from implementation in powerpoint (slide 15/20)
    # This algorithm uses some repairing to help convergence
    new_solution = solution.copy()

    for _ in range(k): # Do k swaps per neighbour
        row_sums = np.sum(COL_ROWS[new_solution], axis=0)
        row_bad = np.where((row_sums < 1) | (row_sums > 1))[0]

        if row_bad.size: # Repair Sequence for invalid solution
            r = np.random.choice(row_bad)

            if row_sums[r] == 0:
                candidates = np.where(COL_ROWS[:, r])[0]
                # Repair - prefer columns that don't introduce overlaps
                safe = candidates[np.all((COL_ROWS[candidates] == 0) | (row_sums == 0), axis=1)] # safe means for all rows, either the column or row_sum is zero, stopping overlaps
                pick_from = safe if safe.size else candidates
                new_solution[np.random.choice(pick_from)] = True
            else:
                # Repair - remove random overlapping column
                selected_covering = np.where(new_solution & COL_ROWS[:, r])[0]
                new_solution[np.random.choice(selected_covering)] = False
        
        else: # If our solution is valid, remove a random column & then repair
            r = np.random.randint(0, COL_ROWS.shape[1]) # get random row
            j_off = np.where(new_solution & COL_ROWS[:, r])[0][0]
            new_solution[j_off] = False

            ## Repair Sequence - Fix hole made in row with a different column
            candidates = np.where(COL_ROWS[:, r])[0] 
            candidates = candidates[candidates != j_off]
            if candidates.size:
                new_solution[np.random.choice(candidates)] = True
            else:
                new_solution[j_off] = True # If unreplacable, undo the change, as the column is necessary

    return new_solution
# --------------------


# Simulated Annealing Algorithm
def simulated_annealing(x0: np.ndarray|None=None, max_iter: int=10000, t0: float=1000, t1: float=1, zero_w_range: tuple[int,int]=(1000,10000), overlap_w_range: tuple[int,int]=(1000,10000), verbose=False) -> np.ndarray:
    def weights(gen):
        zero_w = zero_w_range[0] + (gen/max_iter) * (zero_w_range[1] - zero_w_range[0])
        overlap_w = overlap_w_range[0] + (gen/max_iter) * (overlap_w_range[1] - overlap_w_range[0])
        return zero_w,overlap_w
    
    zero_w,overlap_w = weights(0) # Set initial weights

    x = x0 if x0 is not None else np.zeros(N_COLS, dtype=bool); fit = fitness(x, zero_w=zero_w, overlap_w=overlap_w)
    best = x.copy(); fit_best = fit

    best_is_feas = False

    t = t0; gen = 0

    alpha = np.pow((t1/t0),(1/max_iter)) # t0 is start temp, t1 is end temp

    def temperature(t: float) -> float:
        return t * alpha

    def probability(fit_new, fit_old, t):
        if fit_new < fit_old: return 1
        else: return np.exp((fit_old - fit_new) / t)

    while gen < max_iter:
        t = temperature(t)
        x_new = neighbour(x, k=4)

        zero_w,overlap_w = weights(gen)

        cost_new = cost(x_new); penalty_new = penalty(x_new, zero_w=zero_w, overlap_w=overlap_w); 
        fit_new = cost_new + penalty_new

        if probability(fit_new, fit, t) > np.random.rand():
            x = x_new; fit = fit_new
        
        fit_best = fitness(best, zero_w=zero_w, overlap_w=overlap_w) # recalculate with updated weights
        
        if not best_is_feas and penalty_new == 0: # Once best_is_feas, any better solution also needs to be feasible
            best_is_feas = True
            best = x.copy(); fit_best = fit; cost_best = cost_new
        elif fit < fit_best and bool(penalty_new) != best_is_feas:
            best = x.copy(); fit_best = fit; cost_best = cost_new

        if gen % 1000 == 0 and verbose:
            print(f"Current Cost: {cost(x):.0f} - Best: {cost_best:.0f} | Current Fit: {fit:.0f} - Best: {fit_best:.0f} | Gen: {gen} | Temperature: {t:.2f} | Penalty Weights: {zero_w:.0f},{overlap_w:.0f} ")

        gen += 1

        if fit_best == cost_best and fit_best <= TARGET_OPTIMAL:
            return best, gen
    return best, gen



import csv

np.random.seed(24)

for f_name in ['sppnw42', 'sppnw43', 'sppnw41']:
    print(f"Beginning Processing {f_name}...")
    
    output_path = f'results/sa_{f_name}_results.csv'

    (N_ROWS,N_COLS), COL_COSTS, COL_ROWS, TARGET_OPTIMAL = load_sppwn(f_name)
    ZERO_W_RANGE = (2000,10000); OVERLAP_W_RANGE = (1000,10000)

    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['trial', 'cost', 'gens', 'feasible', 'num_columns', 'overlaps', 'uncovered', 'columns_used'])

    TRIALS = 30

    for x in range(TRIALS):
        print(f"Current Trial: {x}")
        x0 = np.random.randint(0,2,N_COLS,dtype=bool) # Random init

        best, num_gens = simulated_annealing(x0=x0, max_iter=1000, t0=15000, t1=12, zero_w_range=ZERO_W_RANGE, overlap_w_range=OVERLAP_W_RANGE, verbose=True)

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