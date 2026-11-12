import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
)

def garantir_pastas():
    os.makedirs('outputs/figuras', exist_ok=True)
    os.makedirs('outputs/tabelas', exist_ok=True)

def carregar_e_preparar_dados():
    df = pd.read_csv('data/processed/celula_mes.csv')
    df_encoded = pd.get_dummies(df, columns=['estacao'], drop_first=True)
    
    features = ['grid_x', 'grid_y', 'ano', 'mes', 'centro_x', 'centro_y']
    if 'estacao_seca' in df_encoded.columns:
        features.append('estacao_seca')
        
    return df_encoded, features

def avaliar_cenario(df, features, mask_treino, mask_teste, nome_cenario):
    X_train = df.loc[mask_treino, features]
    y_train = df.loc[mask_treino, 'atropelamento']
    X_test = df.loc[mask_teste, features]
    y_test = df.loc[mask_teste, 'atropelamento']
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    modelos = {
        'MLP_Classifier': MLPClassifier(hidden_layer_sizes=(64, 32), max_iter=500, random_state=42),
        'Random_Forest': RandomForestClassifier(n_estimators=100, random_state=42)
    }
    
    resultados_cenario = []
    
    for nome_modelo, modelo in modelos.items():
        if nome_modelo == 'MLP_Classifier':
            modelo.fit(X_train_scaled, y_train)
            y_pred = modelo.predict(X_test_scaled)
            y_proba = modelo.predict_proba(X_test_scaled)[:, 1]
        else:
            modelo.fit(X_train, y_train)
            y_pred = modelo.predict(X_test)
            y_proba = modelo.predict_proba(X_test)[:, 1]
            
        resultados_cenario.append({
            'Cenário': nome_cenario,
            'Modelo': nome_modelo,
            'Amostras_Treino': len(y_train),
            'Amostras_Teste': len(y_test),
            'Acurácia': round(accuracy_score(y_test, y_pred), 4),
            'Precisão': round(precision_score(y_test, y_pred, zero_division=0), 4),
            'Recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
            'F1-Score': round(f1_score(y_test, y_pred, zero_division=0), 4),
            'ROC-AUC': round(roc_auc_score(y_test, y_proba), 4)
        })
        
    return resultados_cenario

def gerar_grafico_comparativo(df_res):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    metricas = ['F1-Score', 'ROC-AUC']
    
    for idx, metrica in enumerate(metricas):
        pivot = df_res.pivot(index='Cenário', columns='Modelo', values=metrica)
        pivot.plot(kind='bar', ax=axes[idx], rot=0, color=['#1f77b4', '#2ca02c'])
        axes[idx].set_title(f'Comparação de {metrica} por Estratégia de Split')
        axes[idx].set_ylabel(metrica)
        axes[idx].set_ylim(0, 1.0)
        axes[idx].grid(axis='y', linestyle='--', alpha=0.6)
        axes[idx].legend(title='Modelo')
        
    plt.tight_layout()
    plt.savefig('outputs/figuras/comparacao_experimentos_split.png')
    plt.close()

def executar_experimentos():
    garantir_pastas()
    df, features = carregar_e_preparar_dados()
    
    todos_resultados = []
    
    # 1. Split Temporal: Treino (2010-2013) vs Teste (2014-2015)
    mask_treino_temp = df['ano'] <= 2013
    mask_teste_temp = df['ano'] >= 2014
    todos_resultados.extend(
        avaliar_cenario(df, features, mask_treino_temp, mask_teste_temp, 'Split Temporal (2010-13 vs 2014-15)')
    )
    
    # 2. Split Espacial: Treino (Metade Sul UTM Y) vs Teste (Metade Norte UTM Y)
    mediana_y = df['centro_y'].median()
    mask_treino_esp = df['centro_y'] <= mediana_y
    mask_teste_esp = df['centro_y'] > mediana_y
    todos_resultados.extend(
        avaliar_cenario(df, features, mask_treino_esp, mask_teste_esp, 'Split Espacial (Sul vs Norte)')
    )
    
    df_res = pd.DataFrame(todos_resultados)
    df_res.to_csv('outputs/tabelas/metricas_experimentos_split.csv', index=False)
    gerar_grafico_comparativo(df_res)
    
    print("\n=== RESULTADOS DOS EXPERIMENTOS DE GENERALIZAÇÃO (ETAPA 7.1) ===")
    print(df_res.to_string(index=False))
    print("\n[OK] Tabela salva em 'outputs/tabelas/metricas_experimentos_split.csv'")
    print("[OK] Gráfico salvo em 'outputs/figuras/comparacao_experimentos_split.png'")

if __name__ == '__main__':
    executar_experimentos()