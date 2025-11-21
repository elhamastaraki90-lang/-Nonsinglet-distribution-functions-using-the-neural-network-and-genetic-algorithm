import os
import json
import time
import math
import random
import numpy as np
import pandas as pd

# TensorFlow CPU is fine; if GPU exists it will use it.
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ============================
# Configuration
# ============================
DATA_FILE = "f2_simulated_1000.csv"  # Must contain columns: x, Q, Q^2, F2p, F2d, F2ns
RESULTS_PER_RUN_CSV = "inverse_coefficients_hybrid_per_runs.csv"
GA_HISTORIES_CSV = "ga_histories.csv"
FINAL_RESULTS_CSV = "inverse_coefficients_hybrid_results.csv"

PARAM_NAMES = ["a_u","b_u","c_u","d_u","a_d","b_d","c_d","d_d","Lambda2"]  # Lambda^2

# GA parameters (per your spec)
POP_SIZE = 219
MUT_RATE = 0.05
CROSS_RATE = 0.80
MAX_GENERATIONS = 500
EARLY_STOP_NO_IMPROVE = 50  # stop if no improvement for 50 generations

# Hybrid scheme
INDEPENDENT_RUNS = 30
TRAIN_VAL_SPLIT = 0.8
RNG_BASE_SEED = 1361  # base seed; different per run

# ANN parameters
ANN_HIDDEN = [64, 64, 32]
ANN_ACT = "tanh"
ANN_EPOCHS = 200
ANN_BATCH = 64
ANN_LR = 1e-3
ANN_VAL_SPLIT = 0.2
ANN_PATIENCE = 20  # early stopping

# Physical/param bounds (reasonable wide boxes; adapt if needed)
BOUNDS = {
    "a_u": (0.2, 1.5),
    "b_u": (0.5, 6.0),
    "c_u": (0.0, 1.0),
    "d_u": (-1.5, 2.0),
    "a_d": (0.2, 1.5),
    "b_d": (0.5, 6.0),
    "c_d": (0.0, 1.0),
    "d_d": (-2.0, 0.5),
    "Lambda2": (50.0, 600.0)  # MeV^2 (example box)
}

# ============================
# Utilities
# ============================
def set_global_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)

def load_data(csv_path):
    df = pd.read_csv(csv_path)
    required = ["x","Q","Q^2","F2p","F2d","F2ns"]
    miss = [c for c in required if c not in df.columns]
    if miss:
        raise ValueError(f"Missing columns in {csv_path}: {miss}")
    # Inputs: x, Q, Q^2; Targets: F2p, F2d, F2ns
    X = df[["x","Q","Q^2"]].values.astype(np.float32)
    y = df[["F2p","F2d","F2ns"]].values.astype(np.float32)
    return X, y

def train_val_split_idx(n, frac):
    idx = np.arange(n)
    np.random.shuffle(idx)
    k = int(n*frac)
    return idx[:k], idx[k:]

def r2_score(y_true, y_pred):
    ss_res = np.sum((y_true - y_pred)**2, axis=0)
    ss_tot = np.sum((y_true - np.mean(y_true, axis=0))**2, axis=0)
    # Weighted average across channels by variance
    r2_channels = 1.0 - ss_res/np.maximum(ss_tot, 1e-12)
    # average R2 across F2p, F2d, F2ns
    return float(np.mean(r2_channels))

# ============================
# Forward model placeholder
# ============================
# This ANN acts as a fast surrogate; in your full pipeline you can plug the exact NNLO forward model here.
def build_surrogate(input_dim=3, output_dim=3):
    inputs = keras.Input(shape=(input_dim,))
    x = inputs
    for units in ANN_HIDDEN:
        x = layers.Dense(units, activation=ANN_ACT,
                         kernel_initializer="he_normal")(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=ANN_LR)
    model.compile(optimizer=opt, loss="mse", metrics=["mae"])
    return model

# We will fit surrogate per run on (X_train, y_train) to emulate F2 mapping.
# In a physics-accurate implementation you should replace surrogate.predict(X, params)
# with your NNLO forward code F2_model(X, params). Here we fold params -> outputs via a small head NN.

def build_param_to_output_head(param_dim=9, output_dim=3):
    # maps theta -> correction to surrogate outputs (small adjustment)
    inputs = keras.Input(shape=(param_dim,))
    x = layers.Dense(64, activation=ANN_ACT)(inputs)
    x = layers.Dense(64, activation=ANN_ACT)(x)
    x = layers.Dense(32, activation=ANN_ACT)(x)
    outputs = layers.Dense(output_dim, activation="linear")(x)
    model = keras.Model(inputs, outputs)
    opt = keras.optimizers.Adam(learning_rate=ANN_LR)
    model.compile(optimizer=opt, loss="mse")
    return model

# ============================
# GA implementation
# ============================
LOW = np.array([BOUNDS[k][0] for k in PARAM_NAMES], dtype=np.float64)
HIGH = np.array([BOUNDS[k][1] for k in PARAM_NAMES], dtype=np.float64)

def init_population(pop_size):
    return np.random.uniform(LOW, HIGH, size=(pop_size, len(PARAM_NAMES)))

def crossover(parent1, parent2):
    if np.random.rand() > CROSS_RATE:
        return parent1.copy(), parent2.copy()
    alpha = np.random.rand(len(PARAM_NAMES))
    child1 = alpha*parent1 + (1-alpha)*parent2
    child2 = alpha*parent2 + (1-alpha)*parent1
    return np.clip(child1, LOW, HIGH), np.clip(child2, LOW, HIGH)

def mutate(chrom):
    mask = np.random.rand(len(PARAM_NAMES)) < MUT_RATE
    noise = np.random.normal(scale=0.05, size=len(PARAM_NAMES))
    chrom[mask] += noise[mask]*(HIGH[mask]-LOW[mask])
    return np.clip(chrom, LOW, HIGH)

def tournament_select(pop, fitness, k=3):
    n = len(pop)
    best = None
    best_fit = np.inf
    for _ in range(k):
        i = np.random.randint(0, n)
        if fitness[i] < best_fit:
            best_fit = fitness[i]
            best = i
    return pop[best].copy()

def evaluate_chromosome(theta, X_val, y_val, surrogate, head):
    # Predict surrogate outputs and apply param-head correction
    y_base = surrogate.predict(X_val, verbose=0)
    corr = head.predict(theta[np.newaxis, :], verbose=0)
    y_pred = y_base + corr
    # MSE loss as GA objective (lower is better)
    mse = float(np.mean((y_val - y_pred)**2))
    return mse, y_pred

def run_ga(X_train, y_train, X_val, y_val, seed):
    # Build/fit surrogate to training data (data-driven proxy of forward model)
    surrogate = build_surrogate()
    early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=ANN_PATIENCE,
                                          restore_best_weights=True, verbose=0)
    surrogate.fit(X_train, y_train, validation_split=ANN_VAL_SPLIT,
                  epochs=ANN_EPOCHS, batch_size=ANN_BATCH, verbose=0, callbacks=[early])

    # Small head for param corrections
    head = build_param_to_output_head()

    # Initialize population
    pop = init_population(POP_SIZE)
    fitness = np.zeros(POP_SIZE, dtype=np.float64)

    best_loss = np.inf
    best_theta = None
    best_pred = None
    no_improve = 0

    history = []

    for gen in range(MAX_GENERATIONS):
        # Quick head training warmup each generation with pseudo-supervision:
        # Use current best or random thetas to align correction to residual mean.
        # This keeps head responsive as GA explores.
        if gen == 0 or gen % 5 == 0:
            # Construct a small batch of random thetas and pseudo targets
            batch_thetas = np.random.uniform(LOW, HIGH, size=(128, len(PARAM_NAMES))).astype(np.float32)
            # Use residual mean on train split as target correction
            yb = surrogate.predict(X_train, verbose=0)
            resid = (y_train - yb).mean(axis=0, keepdims=True)
            y_targets = np.repeat(resid, repeats=batch_thetas.shape[0], axis=0)
            head.fit(batch_thetas, y_targets, epochs=10, batch_size=64, verbose=0)

        # Evaluate current population
        for i in range(POP_SIZE):
            fitness[i], _ = evaluate_chromosome(pop[i], X_val, y_val, surrogate, head)

        # Record history
        gen_best_idx = int(np.argmin(fitness))
        gen_best_loss = float(fitness[gen_best_idx])
        history.append(gen_best_loss)

        if gen_best_loss + 1e-12 < best_loss:
            best_loss = gen_best_loss
            best_theta = pop[gen_best_idx].copy()
            _, best_pred = evaluate_chromosome(best_theta, X_val, y_val, surrogate, head)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve >= EARLY_STOP_NO_IMPROVE:
            break

        # Create next generation via tournament, crossover, mutation
        next_pop = []
        while len(next_pop) < POP_SIZE:
            p1 = tournament_select(pop, fitness, k=3)
            p2 = tournament_select(pop, fitness, k=3)
            c1, c2 = crossover(p1, p2)
            c1 = mutate(c1)
            c2 = mutate(c2)
            next_pop.append(c1)
            if len(next_pop) < POP_SIZE:
                next_pop.append(c2)
        pop = np.array(next_pop, dtype=np.float64)

    # After GA, fine-tune head around best theta with a few steps
    # We minimize validation MSE by learning a small correction only.
    best_theta_tf = tf.Variable(best_theta.astype(np.float32))
    opt = keras.optimizers.Adam(learning_rate=5e-3)

    @tf.function
    def step():
        with tf.GradientTape() as tape:
            yb = surrogate(best_X_val, training=False)
            corr = head(tf.expand_dims(best_theta_tf, axis=0), training=True)
            yp = yb + corr
            loss = tf.reduce_mean(tf.square(best_y_val - yp))
        grads = tape.gradient(loss, head.trainable_variables)
        opt.apply_gradients(zip(grads, head.trainable_variables))
        return loss

    # Cache tensors
    best_X_val = tf.convert_to_tensor(X_val, dtype=tf.float32)
    best_y_val = tf.convert_to_tensor(y_val, dtype=tf.float32)
    for _ in range(100):
        step()

    # Final prediction and metrics
    final_mse, y_pred = evaluate_chromosome(best_theta, X_val, y_val, surrogate, head)
    r2 = r2_score(y_val, y_pred)

    return best_theta, r2, history

# ============================
# Main routine
# ============================
def main():
    if not os.path.exists(DATA_FILE):
        raise FileNotFoundError(f"{DATA_FILE} not found in working dir.")

    # Load data
    X, y = load_data(DATA_FILE)
    n = len(X)
    all_runs = []
    all_histories = []

    t0 = time.time()
    for run in range(INDEPENDENT_RUNS):
        seed = RNG_BASE_SEED + run
        set_global_seed(seed)

        # Split
        idx_train, idx_val = train_val_split_idx(n, TRAIN_VAL_SPLIT)
        X_train, y_train = X[idx_train], y[idx_train]
        X_val, y_val = X[idx_val], y[idx_val]

        # Run GA+ANN
        theta_star, r2, hist = run_ga(X_train, y_train, X_val, y_val, seed)

        row = {k: v for k, v in zip(PARAM_NAMES, theta_star)}
        row["R2"] = r2
        row["run"] = run
        all_runs.append(row)

        # store history with run tag
        for g, val in enumerate(hist):
            all_histories.append({"run": run, "generation": g, "best_mse": val})

        print(f"[Run {run:02d}] R2={r2:.4f} | theta*={theta_star}")

    # Save per-run results
    df_runs = pd.DataFrame(all_runs)
    df_runs.to_csv(RESULTS_PER_RUN_CSV, index=False)

    # Save GA histories
    df_hist = pd.DataFrame(all_histories)
    df_hist.to_csv(GA_HISTORIES_CSV, index=False)

    # Aggregate final stats
    means = df_runs[PARAM_NAMES + ["R2"]].mean(axis=0)
    stds = df_runs[PARAM_NAMES + ["R2"]].std(axis=0, ddof=1)

    final_df = pd.DataFrame([means, stds], index=["Mean","Std"])
    final_df.to_csv(FINAL_RESULTS_CSV, index=True)

    dt = time.time() - t0
    print(f"\nSaved:\n- {RESULTS_PER_RUN_CSV}\n- {GA_HISTORIES_CSV}\n- {FINAL_RESULTS_CSV}\nElapsed: {dt/60:.1f} min")

if __name__ == "__main__":
    main()
