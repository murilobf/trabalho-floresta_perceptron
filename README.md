# Instruções de execução para Random Forest 

## 1. Treinar, avaliar e salvar dados dos gráficos

```bash
python3 random_forest/random_forest_sinais_vitais.py 02_treino_sinais_vitais_com_label.txt
```

## 2. Previsões (teste cego)

```bash
# arquivo sem label fornecido (atenção: contém os mesmos exemplos do treino, ver artigo_base.md)
python3 random_forest/random_forest_sinais_vitais.py 02_treino_sinais_vitais_com_label.txt \
    --prever 01_treino_sinais_vitais_sem_label.txt
```

## 3. Gráficos

```bash
python3 random_forest/graficos_rf.py
```


### 4. Exploração de hiperparâmetros

```bash
python3 random_forest/explorar_hiperparametros.py 02_treino_sinais_vitais_com_label.txt
```