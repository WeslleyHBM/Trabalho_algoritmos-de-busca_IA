import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, ConfusionMatrixDisplay
)

def garantir_pastas():
    os.makedirs('outputs/figuras', exist_ok=True)
    os.makedirs('outputs/tabelas', exist_ok=True)

def carregar_dados():
    caminho = 'data/processed/celula_mes.csv'
    return pd.read_csv(caminho)

def preparar_features(df):
    # One-Hot Encoding para estacao
    df_encoded = pd.get_dummies(df, columns=['estacao'], drop_first=True)
    
    features = ['grid_x', 'grid_y', 'ano', 'mes', 'centro_x', 'centro_y']
    if 'estacao_seca' in df_encoded.columns:
        features.append('estacao_seca')
        
    X = df_encoded[features]
    y = df_encoded['atropelamento']
    
    return X, y

def treinar_e_avaliar():
    garantir_pastas()
    df = carregar_dados()
    X, y = preparar_features(df)
    
    # Split Treino/Teste (80/20 com estratificação de classe)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Padronização de escala (essencial para convergência da MLP)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelos = {
        'MLP_Classifier': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    resultados = []
    
    for nome, modelo in modelos.items():
        if nome == 'MLP_Classifier':
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]
            
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_proba)
        
        resultados.append({
            'Modelo': nome,
            'Acurácia': round(acc, 4),
            'Precisão': round(prec, 4),
            'Recall': round(rec, 4),
            'F1-Score': round(f1, 4),
            'ROC-AUC': round(auc, 4)
        })
        
        # Gerar e guardar Matriz de Confusão
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Ausência (0)', 'Ocorrência (1)'])
        disp.plot(cmap='Blues')
        plt.title(f'Matriz de Confusão - {nome}')
        plt.tight_layout()
        plt.savefig(f'outputs/figuras/matriz_confusao_{nome.lower()}.png')
        plt.close()
        
    df_res = pd.DataFrame(resultados)
    df_res.to_csv('outputs/tabelas/metricas_classificacao_binaria.csv', index=False)
    
    print("\n=== RESULTADOS DA CLASSIFICAÇÃO BINÁRIA (ETAPA 3.1) ===")
    print(df_res.to_string(index=False))
    print("\n[OK] Gráficos salvos em 'outputs/figuras/' e métricas em 'outputs/tabelas/'.")

if __name__ == '__main__':
    treinar_e_avaliar()