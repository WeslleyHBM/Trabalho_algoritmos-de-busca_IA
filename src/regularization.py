import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, roc_auc_score, log_loss

SEED = 42
np.random.seed(SEED)

def executar_regularizacao():
    print("Iniciando Etapa 6 - Regularização e Análise de Overfitting...")

    # 1. Carregamento dos dados
    df = pd.read_csv('data/processed/celula_mes.csv')
    y = df['atropelamento']
    X = df[['grid_x', 'grid_y', 'ano', 'mes', 'estacao', 'centro_x', 'centro_y']].copy()
    X = pd.get_dummies(X, drop_first=True)

    # 2. Divisão (Treino, Validação e Teste)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=SEED, stratify=y_temp)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 3. MLP SEM Regularização (Baseline)
    print("Treinando MLP Sem Regularização...")
    mlp_sem = MLPClassifier(
        hidden_layer_sizes=(64, 128), 
        activation='relu', 
        solver='adam', 
        learning_rate_init=0.00249, 
        alpha=0.0,
        batch_size=128, 
        early_stopping=False,
        random_state=SEED, 
        max_iter=1 
    )

    epochs = 150
    classes = np.unique(y_train)
    loss_treino_sem, loss_val_sem = [], []

    for _ in range(epochs):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp_sem.partial_fit(X_train_scaled, y_train, classes=classes)
            
        y_train_proba = mlp_sem.predict_proba(X_train_scaled)
        y_val_proba = mlp_sem.predict_proba(X_val_scaled)
        loss_treino_sem.append(log_loss(y_train, y_train_proba))
        loss_val_sem.append(log_loss(y_val, y_val_proba))

    # 4. MLP COM Regularização (L2 + Early Stopping)
    print("Treinando MLP Com Regularização (L2 Weight Decay + Early Stopping)...")
    mlp_com = MLPClassifier(
        hidden_layer_sizes=(64, 128), 
        activation='relu', 
        solver='adam', 
        learning_rate_init=0.00249, 
        alpha=0.000306, # L2 do Optuna
        batch_size=128, 
        early_stopping=True, # Parada antecipada ativada
        validation_fraction=0.15,
        n_iter_no_change=10,
        random_state=SEED, 
        max_iter=150
    )
    mlp_com.fit(X_train_scaled, y_train)

    # 5. Avaliação Final (Conjunto de Teste)
    def avaliar_modelo(modelo, nome):
        pred = modelo.predict(X_test_scaled)
        proba = modelo.predict_proba(X_test_scaled)[:, 1]
        return {
            "Modelo": nome,
            "F1-Score": f1_score(y_test, pred),
            "ROC-AUC": roc_auc_score(y_test, proba)
        }

    resultados = pd.DataFrame([
        avaliar_modelo(mlp_sem, "MLP Sem Regularização"),
        avaliar_modelo(mlp_com, "MLP Com L2 e Early Stopping")
    ])
    
    print("\nResultados no Conjunto de Teste (Dados Inéditos):")
    print(resultados.to_string(index=False))
    resultados.to_csv('outputs/tabelas/metricas_regularizacao.csv', index=False)

    # 6. Geração dos Gráficos
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Curva do modelo SEM regularização (mostra o overfitting claro)
    ax[0].plot(loss_treino_sem, label='Treino')
    ax[0].plot(loss_val_sem, label='Validação')
    ax[0].set_title('Curva de Loss - Sem Regularização')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Log Loss')
    ax[0].legend()

    # Curva do modelo COM regularização (Sklearn guarda o loss.curve_ do modelo inteiro fitado com early_stopping)
    ax[1].plot(mlp_com.loss_curve_, label='Treino')
    if hasattr(mlp_com, 'validation_scores_'):
        ax[1].plot([1 - x for x in mlp_com.validation_scores_], label='Validação (Erro)') # Sklearn guarda score (1-loss)
    ax[1].set_title('Curva de Loss - Com L2 e Early Stopping')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('Loss / Erro de Validação')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('outputs/figuras/comparacao_regularizacao.png')
    plt.close()

    print("\nEtapa 6 concluída! Gráficos e tabela salvos nas pastas 'outputs'.")

if __name__ == "__main__":
    executar_regularizacao()