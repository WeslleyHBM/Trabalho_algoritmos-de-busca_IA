import subprocess
import sys
import os

def run_script(script_name):
    script_path = os.path.join('src', script_name)
    print(f"\n{'='*50}")
    print(f"Executando {script_name}...")
    print(f"{'='*50}\n")
    
    if not os.path.exists(script_path):
        print(f"Erro: O arquivo {script_path} não foi encontrado.")
        return False
        
    try:
        # Executa o script e redireciona a saída para o terminal atual
        result = subprocess.run([sys.executable, script_path], check=True)
        print(f"\n{'='*50}")
        print(f"✅ {script_name} concluído com sucesso!")
        print(f"{'='*50}\n")
        return True
    except subprocess.CalledProcessError as e:
        print(f"\n{'='*50}")
        print(f"❌ Erro ao executar {script_name}. Código de saída: {e.returncode}")
        print(f"{'='*50}\n")
        return False

def main():
    # Lista de scripts na ordem correta de execução
    scripts = [
        "eda.py",
        "eda_complementar.py",
        "preprocessing.py",
        "classification.py",
        "classification_multiclasse.py",  # Baseado no print, parece ter 'classification_...'
        "regression.py",
        "optuna_tuning.py",
        "regularization.py",
        "experiments.py"
    ]
    
    print("Iniciando pipeline do RodoFauna...")
    print("Os scripts serão executados um por um.\n")
    
    for script in scripts:
        sucesso = run_script(script)
        if not sucesso:
            resposta = input("Deseja continuar com o próximo script mesmo assim? (s/n): ")
            if resposta.lower() != 's':
                print("Execução interrompida.")
                break

if __name__ == '__main__':
    main()