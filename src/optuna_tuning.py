import pandas as pd
import numpy as np
import optuna
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import roc_auc_score
import optuna.visualization.matplotlib as vis

# Configurações de reprodutibilidade
SEED = 42
np.random.seed(SEED)

def executar_optuna():
    print("Iniciando Etapa 5 - Otimização de Hiperparâmetros com Optuna...")

    # 1. Carregamento dos dados
    df = pd.read_csv('data/processed/celula_mes.csv')
    
    # Otimizaremos o modelo binário (Prever áreas de risco / hotspots)
    y = df['atropelamento']
    
    colunas_permitidas = ['grid_x', 'grid_y', 'ano', 'mes', 'estacao', 'centro_x', 'centro_y']
    X = df[colunas_permitidas].copy()
    X = pd.get_dummies(X, drop_first=True)

    # 2. Divisão (Treino e Validação EXCLUSIVAMENTE, sem Teste)
    # Primeiro separamos os 20% do teste original para deixá-lo intocável
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    
    # Agora dividimos os 80% restantes em Treino e Validação para o Optuna
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=SEED, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)

    # 3. Função Objetivo
    def objective(trial):
        # Definição do Espaço de Busca
        n_layers = trial.suggest_int('n_layers', 1, 2)
        layers = []
        for i in range(n_layers):
            layers.append(trial.suggest_categorical(f'n_units_l{i}', [32, 64, 128]))
        
        activation = trial.suggest_categorical('activation', ['relu', 'tanh'])
        solver = trial.suggest_categorical('solver', ['adam', 'sgd'])
        lr = trial.suggest_float('learning_rate_init', 1e-4, 1e-2, log=True)
        alpha = trial.suggest_float('alpha', 1e-5, 1e-1, log=True) # Weight decay / Regularização L2
        batch_size = trial.suggest_categorical('batch_size', [32, 64, 128, 256])

        mlp = MLPClassifier(
            hidden_layer_sizes=tuple(layers),
            activation=activation,
            solver=solver,
            learning_rate_init=lr,
            alpha=alpha,
            batch_size=batch_size,
            max_iter=50, # Reduzido para não demorar muito na sua máquina
            random_state=SEED,
            early_stopping=True
        )

        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.fit(X_train_scaled, y_train)

        # A métrica objetivo escolhida foi ROC-AUC, pois a base é desbalanceada
        y_val_proba = mlp.predict_proba(X_val_scaled)[:, 1]
        auc = roc_auc_score(y_val, y_val_proba)
        return auc

    # 4. Execução do Estudo (30 Trials)
    print("Buscando hiperparâmetros (30 trials)... Pode levar cerca de 1 a 2 minutos.")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=30)

    # 5. Salvamento dos Resultados
    print("\n=== Melhores Hiperparâmetros ===")
    print(study.best_params)
    
    with open('outputs/tabelas/optuna_best_params.txt', 'w') as f:
        f.write("=== Melhores Hiperparâmetros (Optuna) ===\n")
        for k, v in study.best_params.items():
            f.write(f"{k}: {v}\n")
        f.write(f"\nMelhor ROC-AUC Validação: {study.best_value:.4f}")

    # 6. Geração de Gráficos (usando matplotlib backend para não exigir bibliotecas extras)
    vis.plot_optimization_history(study)
    plt.title("Histórico de Otimização (Optuna)")
    plt.tight_layout()
    plt.savefig("outputs/figuras/optuna_historico.png")
    plt.close()

    vis.plot_param_importances(study)
    plt.title("Importância dos Hiperparâmetros")
    plt.tight_layout()
    plt.savefig("outputs/figuras/optuna_importancia.png")
    plt.close()

    print("Etapa 5 concluída! Gráficos e txt salvos nas pastas 'outputs'.")

if __name__ == "__main__":
    executar_optuna()