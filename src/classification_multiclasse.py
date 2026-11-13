import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
                             f1_score, confusion_matrix, classification_report, log_loss)

# Configurações de reprodutibilidade
SEED = 42
np.random.seed(SEED)

def executar_classificacao_multiclasse():
    print("Iniciando Etapa 3.2 - Classificação Multiclasse...")

    # 1. Carregamento dos dados
    df = pd.read_csv('data/processed/dados_processados.csv')
    
    # Garantir que estamos usando apenas as 4 classes principais
    classes_alvo = ['aves', 'reptiles', 'mammals', 'amphibians']
    df = df[df['classe'].isin(classes_alvo)].copy()

    # Separação de X e y
    cols_to_drop = ['classe', 'id', 'OBJECTID', 'especie_nm'] 
    X = df.drop(columns=[col for col in cols_to_drop if col in df.columns])
    
    # Tratamento de variáveis categóricas remanescentes em X (One-Hot Encoding)
    X = pd.get_dummies(X, drop_first=True)
    
    y = df['classe']

    # Codificação do Alvo
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    # 2. Divisão dos Dados (Treino: 70%, Validação: 15%, Teste: 15%)
    # Uso do stratify=y_encoded devido ao forte desbalanceamento
    X_temp, X_test, y_temp, y_test = train_test_split(X, y_encoded, test_size=0.15, random_state=SEED, stratify=y_encoded)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.1765, random_state=SEED, stratify=y_temp) # 0.1765 de 85% = 15%

    # 3. Padronização (Ajustada SOMENTE no treino)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    # 4. Treinamento da MLP com extração de métricas por época
    print("Treinando MLP e extraindo curvas de aprendizado...")
    mlp = MLPClassifier(hidden_layer_sizes=(64, 32), activation='relu', solver='adam', random_state=SEED)
    
    train_losses, val_losses = [], []
    train_f1, val_f1 = [], []
    classes = np.unique(y_train)
    
    epochs = 150
    for epoch in range(epochs):
        mlp.partial_fit(X_train_scaled, y_train, classes=classes)
        
        # Loss
        y_train_proba = mlp.predict_proba(X_train_scaled)
        y_val_proba = mlp.predict_proba(X_val_scaled)
        train_losses.append(log_loss(y_train, y_train_proba))
        val_losses.append(log_loss(y_val, y_val_proba))
        
        # F1 Macro
        y_train_pred = mlp.predict(X_train_scaled)
        y_val_pred = mlp.predict(X_val_scaled)
        train_f1.append(f1_score(y_train, y_train_pred, average='macro'))
        val_f1.append(f1_score(y_val, y_val_pred, average='macro'))

    # Avaliação MLP no Teste
    y_test_pred_mlp = mlp.predict(X_test_scaled)
    
    # 5. Treinamento do Modelo Tradicional (Random Forest) para comparação
    print("Treinando Random Forest para comparação...")
    rf = RandomForestClassifier(n_estimators=100, random_state=SEED, class_weight='balanced')
    rf.fit(X_train, y_train) # RF não precisa de dados padronizados
    y_test_pred_rf = rf.predict(X_test)

    # 6. Geração de Métricas
    def extrair_metricas(y_true, y_pred, modelo_nome):
        return {
            'Modelo': modelo_nome,
            'Accuracy': accuracy_score(y_true, y_pred),
            'Precision Macro': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'Recall Macro': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'F1 Macro': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }

    resultados = [
        extrair_metricas(y_test, y_test_pred_mlp, 'MLP Multiclasse'),
        extrair_metricas(y_test, y_test_pred_rf, 'Random Forest Multiclasse')
    ]
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv('outputs/tabelas/metricas_classificacao_multiclasse.csv', index=False)
    print("\nResultados Salvos:\n", df_resultados)

    # 7. Geração de Gráficos (Curvas da MLP)
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    
    # Loss Curve
    ax[0].plot(train_losses, label='Treino')
    ax[0].plot(val_losses, label='Validação')
    ax[0].set_title('Curva de Loss por Época (MLP)')
    ax[0].set_xlabel('Épocas')
    ax[0].set_ylabel('Log Loss')
    ax[0].legend()

    # F1 Macro Curve
    ax[1].plot(train_f1, label='Treino')
    ax[1].plot(val_f1, label='Validação')
    ax[1].set_title('Evolução do F1-Macro por Época (MLP)')
    ax[1].set_xlabel('Épocas')
    ax[1].set_ylabel('F1-Macro')
    ax[1].legend()

    plt.tight_layout()
    plt.savefig('outputs/figuras/curvas_aprendizado_mlp_multiclasse.png')
    plt.close()

    # 8. Matrizes de Confusão
    fig, ax = plt.subplots(1, 2, figsize=(14, 5))
    sns.heatmap(confusion_matrix(y_test, y_test_pred_mlp), annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax[0])
    ax[0].set_title('Matriz de Confusão - MLP')
    ax[0].set_xlabel('Predito')
    ax[0].set_ylabel('Real')

    sns.heatmap(confusion_matrix(y_test, y_test_pred_rf), annot=True, fmt='d', cmap='Greens',
                xticklabels=le.classes_, yticklabels=le.classes_, ax=ax[1])
    ax[1].set_title('Matriz de Confusão - Random Forest')
    ax[1].set_xlabel('Predito')
    ax[1].set_ylabel('Real')

    plt.tight_layout()
    plt.savefig('outputs/figuras/matrizes_confusao_multiclasse.png')
    plt.close()

    print("Etapa 3.2 concluída com sucesso! Gráficos e tabelas salvos nas pastas 'outputs'.")

if __name__ == "__main__":
    executar_classificacao_multiclasse()