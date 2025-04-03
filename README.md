# Estudos de Casos de Dinâmica de Partículas

## Simulação de Pêndulo Elástico

Simulação de Pêndulo Elástico em diversas condições iniciais para observação de seu comportamento.

## Simulação de Pêndulo Duplo

Simulação de Pêndulo Duplo em diversas condições iniciais para observação de seu comportamento.

## Detalhes das Simulações

### Pêndulo Elástico
A simulação do pêndulo elástico utiliza as equações de Lagrange para modelar o movimento de uma massa conectada a uma mola e sujeita à gravidade. O sistema é resolvido numericamente utilizando o método `odeint` da biblioteca SciPy. A simulação permite:
- Visualizar as variações de energia cinética, potencial e total ao longo do tempo.
- Analisar as funções θ(t), dθ(t)/dt, u(t) e du(t)/dt.
- Observar as trajetórias em coordenadas cartesianas (x, y).
- Estudar as fases θ(t) x dθ(t)/dt, u(t) x du(t)/dt e θ(t) x u(t).
- Gerar uma animação do movimento do pêndulo.

### Pêndulo Duplo
A simulação do pêndulo duplo modela o comportamento de dois pêndulos conectados, considerando as equações de Lagrange para descrever o sistema. O movimento é resolvido numericamente e permite:
- Comparar o comportamento do sistema sob diferentes condições gravitacionais (ex.: Terra e Urano).
- Visualizar as posições cartesianas (x, y) dos pêndulos ao longo do tempo.
- Analisar as trajetórias e posições do pêndulo secundário em função do tempo.
- Gerar uma animação do movimento do sistema.

### Requisitos
- Python 3.10 ou superior
- Bibliotecas: `numpy`, `scipy`, `matplotlib`, `sympy`

### Certifique-se de ter todas as dependências instaladas:
    ```bash
    pip install -r requirements.txt
    ```

### Resultados
As simulações geram gráficos e animações que podem ser salvos como arquivos `.gif` ou `.png` para análise posterior.