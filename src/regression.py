import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPRegressor
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Configurações de reprodutibilidade
SEED = 42
np.random.seed(SEED)

def executar_regressao():
    print("Iniciando Etapa 4 - Regressão...")

  
    df = pd.read_csv('data/processed/celula_mes.csv')


    y = df['contagem']
    
 
    colunas_permitidas = ['grid_x', 'grid_y', 'ano', 'mes', 'estacao', 'centro_x', 'centro_y']
    X = df[colunas_permitidas].copy()

  
    X = pd.get_dummies(X, drop_first=True)

    # 3. Divisão Treino/Validação/Teste
    # Na regressão espacial, usamos split aleatório base
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.20, random_state=SEED)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.20, random_state=SEED)

    # 4. Padronização (ajustada somente no treino)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 5. Modelagem - Regressão Linear (Tradicional)
    print("Treinando Regressão Linear...")
    lr = LinearRegression()
    lr.fit(X_train_scaled, y_train)
    y_test_pred_lr = lr.predict(X_test_scaled)

    # 6. Modelagem - Random Forest Regressor (Tradicional)
    print("Treinando Random Forest Regressor...")
    rf = RandomForestRegressor(n_estimators=100, random_state=SEED)
    rf.fit(X_train, y_train) # Árvores não precisam de padronização
    y_test_pred_rf = rf.predict(X_test)

    # 7. Modelagem - MLP Regressor
    print("Treinando MLP Regressor com monitoramento por época...")
    mlp = MLPRegressor(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', 
                       learning_rate_init=0.001, random_state=SEED, max_iter=1)
    
    epochs = 150
    train_losses, val_losses = [], []
    
    # Treinamento iterativo para extração manual das curvas de loss de validação (Sklearn só guarda a de treino no clf.loss_curve_)
    for epoch in range(epochs):
        # catch warnings de max_iter na partial_fit
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            mlp.partial_fit(X_train_scaled, y_train)
            
        y_train_pred_iter = mlp.predict(X_train_scaled)
        y_val_pred_iter = mlp.predict(X_val_scaled)
        
        # O loss tradicional do regressão (MSE)
        train_losses.append(mean_squared_error(y_train, y_train_pred_iter))
        val_losses.append(mean_squared_error(y_val, y_val_pred_iter))

    y_test_pred_mlp = mlp.predict(X_test_scaled)

    # 8. Extração e Salvamento das Métricas
    def calcular_metricas(y_true, y_pred, modelo):
        return {
            'Modelo': modelo,
            'MAE': mean_absolute_error(y_true, y_pred),
            'MSE': mean_squared_error(y_true, y_pred),
            'RMSE': np.sqrt(mean_squared_error(y_true, y_pred)),
            'R2': r2_score(y_true, y_pred)
        }

    metricas = [
        calcular_metricas(y_test, y_test_pred_lr, 'Regressão Linear'),
        calcular_metricas(y_test, y_test_pred_rf, 'Random Forest Regressor'),
        calcular_metricas(y_test, y_test_pred_mlp, 'MLP Regressor')
    ]
    
    df_metricas = pd.DataFrame(metricas)
    df_metricas.to_csv('outputs/tabelas/metricas_regressao.csv', index=False)
    print("\nResultados Salvos:\n", df_metricas)

    # 9. Geração dos Gráficos
    
    # Gráfico 1: Curva de Loss da MLP
    plt.figure(figsize=(8, 5))
    plt.plot(train_losses, label='Treino')
    plt.plot(val_losses, label='Validação')
    plt.title('Curva de Erro Quadrático Médio (MSE) por Época - MLP Regressor')
    plt.xlabel('Épocas')
    plt.ylabel('MSE')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/figuras/regressao_curva_loss_mlp.png')
    plt.close()

    # Gráfico 2: Valores Reais vs. Preditos (MLP e RF)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].scatter(y_test, y_test_pred_rf, alpha=0.3, color='green')
    ax[0].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax[0].set_title('Reais vs Preditos (Random Forest)')
    ax[0].set_xlabel('Quantidade Real')
    ax[0].set_ylabel('Quantidade Predita')

    ax[1].scatter(y_test, y_test_pred_mlp, alpha=0.3, color='blue')
    ax[1].plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--')
    ax[1].set_title('Reais vs Preditos (MLP)')
    ax[1].set_xlabel('Quantidade Real')
    ax[1].set_ylabel('Quantidade Predita')
    
    plt.tight_layout()
    plt.savefig('outputs/figuras/regressao_reais_vs_preditos.png')
    plt.close()

    # Gráfico 3: Gráfico de Resíduos (Erro Real - Predito)
    residuos_rf = y_test - y_test_pred_rf
    residuos_mlp = y_test - y_test_pred_mlp

    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    ax[0].scatter(y_test_pred_rf, residuos_rf, alpha=0.3, color='green')
    ax[0].axhline(y=0, color='r', linestyle='--')
    ax[0].set_title('Resíduos (Random Forest)')
    ax[0].set_xlabel('Quantidade Predita')
    ax[0].set_ylabel('Resíduo')

    ax[1].scatter(y_test_pred_mlp, residuos_mlp, alpha=0.3, color='blue')
    ax[1].axhline(y=0, color='r', linestyle='--')
    ax[1].set_title('Resíduos (MLP)')
    ax[1].set_xlabel('Quantidade Predita')
    ax[1].set_ylabel('Resíduo')

    plt.tight_layout()
    plt.savefig('outputs/figuras/regressao_residuos.png')
    plt.close()

    print("Etapa 4 concluída! Gráficos salvos em 'outputs/figuras/'.")

if __name__ == "__main__":
    executar_regressao()