

import pandas as pd
import numpy as np
from faker import Faker
import random

# Configuração inicial
fake = Faker('pt_BR')
np.random.seed(42)
random.seed(42)

# Gerando dados
n = 1000  # número de registros

data = {
    'id_medidor': [fake.unique.bothify(text='MED-####-??') for _ in range(n)],
    'consumo_kwh': np.round(np.abs(np.random.normal(300, 150, n)), 2),
    'consumo_medio_historico': np.round(np.abs(np.random.normal(300, 120, n)), 2),
    'horas_uso_diarias': np.round(np.abs(np.random.normal(12, 6, n)), 1),
    'dias_sem_leitura': np.random.randint(0, 15, n),
    'tensao_media': np.round(np.random.uniform(210, 240, n), 1),
    'indice_perdas': np.round(np.random.uniform(0.5, 5.0, n), 2),
    'localizacao': np.random.choice(['residencial', 'comercial', 'industrial'], n, p=[0.7, 0.25, 0.05]),
}

# Calculando variação de consumo
data['variação_consumo'] = np.round(((data['consumo_kwh'] - data['consumo_medio_historico']) / 
                                    data['consumo_medio_historico']) * 100, 2)

# Adicionando fraudes (5% dos dados)
fraude = np.zeros(n)
for i in range(n):
    # Padrões suspeitos que indicam possível fraude
    if (data['variação_consumo'][i] < -60 and 
        data['dias_sem_leitura'][i] > 5 and 
        data['tensao_media'][i] < 215):
        fraude[i] = 1
    elif (data['consumo_kwh'][i] < 50 and 
          data['horas_uso_diarias'][i] > 8 and 
          data['localizacao'][i] != 'industrial'):
        fraude[i] = 1
    elif (data['indice_perdas'][i] > 4.5 and 
          data['variação_consumo'][i] < -30):
        fraude[i] = 1

# Adicionando alguns falsos positivos para tornar o problema mais realista
for _ in range(int(n*0.03)):
    idx = random.randint(0, n-1)
    fraude[idx] = 0  # padrão suspeito mas não é fraude

data['fraude'] = fraude.astype(int)

# Criando DataFrame
df = pd.DataFrame(data)

# Salvando para CSV
df.to_csv('dados_medidores_energia.csv', index=False)