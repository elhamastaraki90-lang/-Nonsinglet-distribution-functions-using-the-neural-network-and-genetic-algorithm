# inverse_dglap_nnlo_hybrid_GA_ANN.py
import numpy as np
import pandas as pd
import mpmath as mp
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import time
from typing import Tuple

# ============ Config ============
SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)
mp.mp.dps = 50  # high precision

# GA hyperparameters (per user spec)
POP_SIZE = 219
MUT_RATE = 0.05
CROSS_OVER_RATE = 0.8
MAX_GENERATIONS = 2000
PATIENCE_NO_IMPROV = 50  # stop if no improvement for 50 generations
ELITE_FRAC = 0.12

# ANN hyperparameters (per user spec)
H1, H2, H3 = 64, 64, 32
ACT = 'tanh'
LR = 1e-3
EPOCHS_FINE = 100
BATCH = 64

# Bounds for parameters: au,bu,cu,du, ad,bd,cd,dd, LAMBDA2
BOUNDS = np.array([
    [0.2, 2.0],   # au
    [2.0, 7.0],   # bu
    [-1.0, 1.0],  # cu
    [-2.0, 3.0],  # du
    [0.2, 2.0],   # ad
    [2.0, 7.0],   # bd
    [-1.0, 1.0],  # cd
    [-3.0, 3.0],  # dd
    [0.01, 1.0],  # LAMBDA2 (GeV^2)
])

PARAM_NAMES = ["au","bu","cu","du","ad","bd","cd","dd","LAMBDA2"]

# ============ Data ============
def load_data(csv_path: str):
    df = pd.read_csv(csv_path)
    for col in ["x","Q^2","F2p","F2d","F2ns"]:
        if col not in df.columns:
            raise ValueError(f"Missing column '{col}' in CSV.")
    x = df["x"].values.astype(float)
    Q2 = df["Q^2"].values.astype(float)
    Q = np.sqrt(Q2)
    F2p = df["F2p"].values.astype(float)
    F2d = df["F2d"].values.astype(float)
    F2ns = df["F2ns"].values.astype(float)
    return x, Q, Q2, F2p, F2d, F2ns

# ============ NNLO Model ============
def alpha_s(Q: float, F: int, LAMBDA2: float) -> float:
    LAMBDA = np.sqrt(LAMBDA2)
    beta0 = 11.0 - 2.0 * F / 3.0
    beta1 = 102.0 - 38.0 * F / 3.0
    beta2 = 1428.5 - 5033.0 * F / 18.0 + 325.0 * F**2 / 54.0
    Q2 = Q**2
    L = np.log(Q2 / LAMBDA2)
    a0 = 1.0 / (beta0 * L)
    c1 = -beta1 / (beta0**2) * np.log(L) / L
    c2 = (beta1**2 / (beta0**4)) * ((np.log(L)**2) - np.log(L) - 1.0 + beta2 * beta0 / beta1**2) / (L**0)
    return float(a0 * (1.0 + c1 + c2))

def UVnDA1(n, au, bu, cu, du):
    n = mp.mpf(n)
    au = mp.mpf(au); bu = mp.mpf(bu); cu = mp.mpf(cu); du = mp.mpf(du)
    num = 2 * mp.gamma(1 + bu) * (
        mp.gamma(-1 + au + n) / mp.gamma(au + bu + n)
        + cu * mp.gamma(-0.5 + au + n) / mp.gamma(0.5 + au + bu + n)
        + du * mp.gamma(au + n) / mp.gamma(1 + au + bu + n)
    )
    den = mp.gamma(1 + bu) * (
        mp.gamma(au) / mp.gamma(1 + au + bu)
        + cu * mp.gamma(0.5 + au) / mp.gamma(1.5 + au + bu)
        + du * mp.gamma(1 + au) / mp.gamma(2 + au + bu)
    )
    return num / den

def DVnDA1(n, ad, bd, cd, dd):
    n = mp.mpf(n)
    ad = mp.mpf(ad); bd = mp.mpf(bd); cd = mp.mpf(cd); dd = mp.mpf(dd)
    num = mp.gamma(1 + bd) * (
        mp.gamma(-1 + ad + n) / mp.gamma(ad + bd + n)
        + cd * mp.gamma(-0.5 + ad + n) / mp.gamma(0.5 + ad + bd + n)
        + dd * mp.gamma(ad + n) / mp.gamma(1 + ad + bd + n)
    )
    den = mp.gamma(1 + bd) * (
        mp.gamma(ad) / mp.gamma(1 + ad + bd)
        + cd * mp.gamma(0.5 + ad) / mp.gamma(1.5 + ad + bd)
        + dd * mp.gamma(1 + ad) / mp.gamma(2 + ad + bd)
    )
    return num / den

def mnsNNLO(N, Q, F, LAMBDA2):
    aQ = mp.mpf(alpha_s(Q, F, LAMBDA2))
    aQ0 = mp.mpf(alpha_s(4.0, F, LAMBDA2))
    beta0 = mp.mpf(11) - 2 * F / 3
    beta1 = mp.mpf(102) - 38 * F / 3
    beta2 = mp.mpf("1428.5") - mp.mpf("5033") * F / 18 + mp.mpf(325) * F**2 / 54
    psi = mp.digamma(N + 1) + mp.euler
    P0 = (4/3) * (-4 * psi + 3 + 2 / (N * (N + 1)))
    P1 = (4/3)**2 * (-3 * psi**2 + 4 * mp.zeta(2) - mp.mpf("6.5"))
    P2 = (4/3)*3 * (-8 * psi**3 + 12 * mp.zeta(3) - mp.mpf("9.2"))
    L_a = mp.log(aQ / aQ0)
    E = (P0 / beta0) * L_a \
        + ((P1 * beta0 - P0 * beta1) / beta0**2) * (aQ - aQ0) \
        + ((P2 * beta0**2 - P1 * beta0 * beta1 + P0 * (beta1**2 - beta0 * beta2)) / beta0**3) * (aQ**2 - aQ0**2)
    return mp.e**E

def F2_model(params, x, Q):
    au, bu, cu, du, ad, bd, cd, dd, LAMBDA2 = params
    F = 4  # active flavors
    N_min, N_max = 2, 10
    U_sum, D_sum = mp.mpf(0), mp.mpf(0)
    for N in range(N_min, N_max + 1):
        U_sum += UVnDA1(N, au, bu, cu, du) * mnsNNLO(N, Q, F, LAMBDA2)
        D_sum += DVnDA1(N, ad, bd, cd, dd) * mnsNNLO(N, Q, F, LAMBDA2)
    # Charges: u(2/3)^2=4/9, d(-1/3)^2=1/9
    return float(mp.re((4.0 / 9.0) * U_sum + (1.0 / 9.0) * D_sum))

def F2_channels_model(params, x_arr, Q_arr):
    # returns F2p, F2d, F2ns model arrays (here: p ~ F2, d ~ close proxy, ns = p - d)
    F2p_model = np.array([F2_model(params, x_arr[i], Q_arr[i]) for i in range(len(x_arr))], dtype=float)
    # A simple deuteron proxy (reduce slightly to mimic nuclear effects)
    F2d_model = 0.97 * F2p_model
    F2ns_model = F2p_model - F2d_model
    return F2p_model, F2d_model, F2ns_model

# ============ Loss ============
def loss_mse(params, x, Q, F2p, F2d, F2ns, idx):
    # bounds check
    for i, (lo, hi) in enumerate(BOUNDS):
        if not (lo <= params[i] <= hi):
            return 1e9
    try:
        F2p_m, F2d_m, F2ns_m = F2_channels_model(params, x[idx], Q[idx])
    except Exception:
        return 1e9
    w_p = w_d = w_ns = 1.0
    mse = np.mean(w_p*(F2p_m - F2p[idx])**2 + w_d*(F2d_m - F2d[idx])**2 + w_ns*(F2ns_m - F2ns[idx])**2)
    return float(mse)

# ============ GA ============
rng = np.random.default_rng(SEED)

def random_params():
    return np.array([rng.uniform(lo, hi) for lo, hi in BOUNDS], dtype=float)

def mutate(ind, sigma=0.1):
    child = ind.copy()
    for i in range(len(child)):
        if rng.random() < MUT_RATE:
            span = (BOUNDS[i][1] - BOUNDS[i][0])
            child[i] += rng.normal(0.0, sigma * span)
            child[i] = np.clip(child[i], BOUNDS[i][0], BOUNDS[i][1])
    return child

def crossover(p1, p2):
    if rng.random() < CROSS_OVER_RATE:
        alpha = rng.random()
        child = alpha * p1 + (1 - alpha) * p2
        return child
    return p1.copy()

def genetic_optimize(x, Q, F2p, F2d, F2ns, train_idx, pop_size=POP_SIZE, elite_frac=ELITE_FRAC):
    population = [random_params() for _ in range(pop_size)]
    n_elite = max(1, int(elite_frac * pop_size))
    best_loss = np.inf
    best = None
    no_improv = 0
    history = []

    def score_pop(pop):
        return np.array([loss_mse(ind, x, Q, F2p, F2d, F2ns, train_idx) for ind in pop], dtype=float)

    scores = score_pop(population)
    for gen in range(1, MAX_GENERATIONS + 1):
        order = np.argsort(scores)
        population = [population[i] for i in order]
        scores = scores[order]
        history.append(scores[0])
        if scores[0] + 1e-10 < best_loss:
            best_loss = scores[0]
            best = population[0].copy()
            no_improv = 0
        else:
            no_improv += 1

        # print(f"Gen {gen} | best loss {scores[0]:.6e}")
        if no_improv >= PATIENCE_NO_IMPROV:
            break

        # elitism
        new_pop = population[:n_elite]
        # fill rest
        while len(new_pop) < pop_size:
            i1, i2 = rng.choice(n_elite, 2, replace=False)
            child = crossover(population[i1], population[i2])
            child = mutate(child, sigma=0.06)
            new_pop.append(child)
        population = new_pop
        scores = score_pop(population)

    return best, best_loss, np.array(history)

# ============ ANN ============
def build_ann():
    model = models.Sequential([
        layers.Input(shape=(5,)),  # x, Q2, F2p, F2d, F2ns
        layers.Dense(H1, activation=ACT, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.Dense(H2, activation=ACT, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.Dense(H3, activation=ACT, kernel_regularizer=tf.keras.regularizers.l2(1e-5)),
        layers.Dense(9, activation='linear')
    ])
    opt = tf.keras.optimizers.Adam(learning_rate=LR)
    model.compile(optimizer=opt, loss='mse')
    return model

def params_to_targets(params, x, Q, F2p, F2d, F2ns, idx):
    # ANN supervised targets: constant vector params for each training sample
    Y = np.tile(params, (len(idx), 1))
    X = np.stack([x[idx], Q[idx]**2, F2p[idx], F2d[idx], F2ns[idx]], axis=1)
    return X.astype(np.float32), Y.astype(np.float32)

# ============ Main 30 runs ============
def main():
    x, Q, Q2, F2p, F2d, F2ns = load_data("f2_simulated_1000.csv")

    N = len(x)
    idx_all = np.arange(N)
    rng.shuffle(idx_all)
    split = int(0.8 * N)
    train_idx = idx_all[:split]
    val_idx = idx_all[split:]

    run_params = []
    run_stats = []
    val_losses = []
    histories = []

    start = time.time()

    for run in range(30):
        # GA search
        best_ga_params, best_ga_loss, hist = genetic_optimize(x, Q, F2p, F2d, F2ns, train_idx)
        histories.append(hist)

        # Prepare ANN training data using GA params as supervision
        X_train, Y_train = params_to_targets(best_ga_params, x, Q, F2p, F2d, F2ns, train_idx)
        X_val, Y_val = params_to_targets(best_ga_params, x, Q, F2p, F2d, F2ns, val_idx)

        # Build & train ANN
        ann = build_ann()
        es = callbacks.EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=True)
        ann.fit(X_train, Y_train, validation_data=(X_val, Y_val),
                epochs=EPOCHS_FINE, batch_size=BATCH, verbose=0, callbacks=[es])

        # Predict params on validation set
        Y_pred_val = ann.predict(X_val, verbose=0)
        # Aggregate predicted params over validation to single vector by mean
        params_val_mean = Y_pred_val.mean(axis=0)

        # Evaluate physics loss on validation with predicted params
        val_loss_mse = loss_mse(params_val_mean, x, Q, F2p, F2d, F2ns, val_idx)

        run_params.append(params_val_mean)
        run_stats.append(best_ga_loss)
        val_losses.append(val_loss_mse)

        print(f"Run {run+1:02d}/30 | GA best train-loss={best_ga_loss:.6e} | VAL phys-loss={val_loss_mse:.6e}")

    end = time.time()
    print(f"Total time: {(end-start)/60:.2f} min")

    run_params = np.array(run_params)  # shape (30, 9)
    means = run_params.mean(axis=0)
    stds = run_params.std(axis=0)

    # Print results
    print("\nAveraged coefficients (30 runs):")
    for n, m, s in zip(PARAM_NAMES, means, stds):
        print(f"{n:8s} = {m:.6f} ± {s:.6f}")

    # Save CSV: mean and std
    out = pd.DataFrame([means, stds], index=["Mean", "Std"], columns=PARAM_NAMES)
    out.to_csv("inverse_coefficients_hybrid_results.csv", index=True)

    # Also save per-run params and losses for transparency
    per_run = pd.DataFrame(run_params, columns=PARAM_NAMES)
    per_run["GA_train_best_loss"] = np.array(run_stats)
    per_run["VAL_phys_loss"] = np.array(val_losses)
    per_run.to_csv("inverse_coefficients_hybrid_per_runs.csv", index=False)

    # Save GA histories
    max_len = max(len(h) for h in histories)
    hist_mat = np.full((30, max_len), np.nan)
    for i, h in enumerate(histories):
        hist_mat[i, :len(h)] = h
    pd.DataFrame(hist_mat).to_csv("ga_histories.csv", index=False)

if __name__ == "__main__":
    main()
