import matplotlib.pyplot as plt
import math
import numpy as np

amplitude = 1
frequencia_angular = 1

def mhs(amplitude, frequencia_angular, tempo):
  """
  Representa o Movimento Harmônico Simples no Movimento Circular Uniforme.

  Args:
    amplitude: Amplitude do movimento.
    frequencia_angular: Frequência angular do movimento.
    tempo: Tempo.

  Returns:
    Posição da partícula em relação ao ponto de equilíbrio.
  """

  return amplitude * np.cos(frequencia_angular * tempo)

tempo = np.linspace(0, 30, 1000)

x = mhs(amplitude, frequencia_angular, tempo)

plt.plot(tempo, x)
plt.xlabel("Tempo (s)")
plt.ylabel("Posição (cm)")
plt.show()
