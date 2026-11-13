import os
import time
import random
import warnings
import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
import optuna.visualization.matplotlib as vis
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

# Configurações de reprodutibilidade total
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def avaliar_modelo(nome, modelo, X_tr, y_tr, X_te, y_te):
    inicio = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        modelo.fit(X_tr, y_tr)
    tempo_treino = time.time() - inicio

    y_pred = modelo.predict(X_te)
    y_proba = modelo.predict_proba(X_te)[:, 1]

    return {
        "Modelo": nome,
        "Acurácia": accuracy_score(y_te, y_pred),
        "Precisão": precision_score(y_te, y_pred, zero_division=0),
        "Recall": recall_score(y_te, y_pred, zero_division=0),
        "F1-Score": f1_score(y_te, y_pred, zero_division=0),
        "ROC-AUC": roc_auc_score(y_te, y_proba),
        "Tempo (s)": tempo_treino,
    }


def executar_optuna():
    print("Iniciando Etapa 5 - Otimização de Hiperparâmetros com Optuna...")

    os.makedirs("outputs/tabelas", exist_ok=True)
    os.makedirs("outputs/figuras", exist_ok=True)

    # 1. Carregamento dos dados
    df = pd.read_csv("data/processed/celula_mes.csv")
    y = df["atropelamento"]

    colunas_permitidas = [
        "grid_x",
        "grid_y",
        "ano",
        "mes",
        "estacao",
        "centro_x",
        "centro_y",
    ]
    X = pd.get_dummies(df[colunas_permitidas].copy(), drop_first=True)

    # 2. Divisão (Treino e Validação EXCLUSIVAMENTE na busca; Teste intocável)
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.20, random_state=SEED, stratify=y
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.20, random_state=SEED, stratify=y_temp
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Função Objetivo
    def objective(trial):
        n_layers = trial.suggest_int("n_layers", 1, 2)
        layers = [
            trial.suggest_categorical(f"n_units_l{i}", [32, 64, 128])
            for i in range(n_layers)
        ]

        activation = trial.suggest_categorical("activation", ["relu", "tanh"])
        solver = trial.suggest_categorical("solver", ["adam", "sgd"])
        lr = trial.suggest_float("learning_rate_init", 1e-4, 1e-2, log=True)
        alpha = trial.suggest_float("alpha", 1e-5, 1e-1, log=True)
        batch_size = trial.suggest_categorical("batch_size", [32, 64, 128, 256])

        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(layers),
            activation=activation,
            solver=solver,
            learning_rate_init=lr,
            alpha=alpha,
            batch_size=batch_size,
            max_iter=50,
            random_state=SEED,
            early_stopping=True,
        )

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_train_scaled, y_train)

        y_val_proba = mlp.predict_proba(X_val_scaled)[:, 1]
        return roc_auc_score(y_val, y_val_proba)

    # 4. Execução do Estudo com Sampler Determinístico (30 Trials)
    print("Buscando hiperparâmetros (30 trials)...")
    sampler = optuna.samplers.TPESampler(seed=SEED)
    study = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(objective, n_trials=30)

    best = study.best_params
    best_layers = tuple(best[f"n_units_l{i}"] for i in range(best["n_layers"]))

    # 5. Avaliação Comparativa Antes vs. Depois no Conjunto de Teste
    scaler_final = StandardScaler()
    X_temp_scaled = scaler_final.fit_transform(X_temp)
    X_test_scaled = scaler_final.transform(X_test)

    mlp_baseline = MLPClassifier(
        hidden_layer_sizes=(64, 32),
        max_iter=200,
        random_state=SEED,
    )
    mlp_otimizada = MLPClassifier(
        hidden_layer_sizes=best_layers,
        activation=best["activation"],
        solver=best["solver"],
        learning_rate_init=best["learning_rate_init"],
        alpha=best["alpha"],
        batch_size=best["batch_size"],
        max_iter=200,
        random_state=SEED,
        early_stopping=True,
    )

    resultados = pd.DataFrame(
        [
            avaliar_modelo(
                "MLP_Baseline", mlp_baseline, X_temp_scaled, y_temp, X_test_scaled, y_test
            ),
            avaliar_modelo(
                "MLP_Optuna", mlp_otimizada, X_temp_scaled, y_temp, X_test_scaled, y_test
            ),
        ]
    )

    print("\n=== Melhores Hiperparâmetros ===")
    print(best)
    print(f"Melhor ROC-AUC Validação: {study.best_value:.4f}")
    print("\n=== Comparação no Teste (Antes vs. Depois) ===")
    print(resultados.to_string(index=False))

    with open("outputs/tabelas/optuna_best_params.txt", "w", encoding="utf-8") as f:
        f.write("=== Melhores Hiperparâmetros (Optuna) ===\n")
        for k, v in best.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nMelhor ROC-AUC Validação: {study.best_value:.4f}\n\n")
        f.write("=== Comparação no Teste (Antes vs. Depois) ===\n")
        f.write(resultados.to_string(index=False))

    # 6. Geração de Gráficos Determinísticos
    ax1 = vis.plot_optimization_history(study)
    ax1.figure.suptitle("")
    ax1.set_title("", loc="left")
    ax1.set_title("", loc="right")
    ax1.set_title("Histórico de Otimização (Optuna)", loc="center")
    ax1.set_xlabel("Tentativa (Trial)")
    ax1.set_ylabel("ROC-AUC (Validação)")
    plt.tight_layout()
    plt.savefig("outputs/figuras/optuna_historico.png", dpi=300, bbox_inches="tight")
    plt.close()

    evaluator = optuna.importance.FanovaImportanceEvaluator(seed=SEED)
    ax2 = vis.plot_param_importances(study, evaluator=evaluator)
    ax2.figure.suptitle("")
    ax2.set_title("", loc="left")
    ax2.set_title("", loc="right")
    ax2.set_title("Importância dos Hiperparâmetros", loc="center")
    ax2.set_xlabel("Importância")
    ax2.set_ylabel("Hiperparâmetro")
    plt.tight_layout()
    plt.savefig("outputs/figuras/optuna_importancia.png", dpi=300, bbox_inches="tight")
    plt.close()

    print("\nEtapa 5 concluída! Gráficos e txt salvos nas pastas 'outputs'.")


if __name__ == "__main__":
    executar_optuna()