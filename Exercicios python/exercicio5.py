# -*- coding: utf-8 -*-
"""Untitled0.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1WnbA7h_JpMZxf6EUxigHb6dnUaSZp_RS
"""

#Entrada
n1 = float(input("Digite a nota 1: "))
n2 = float(input("Digite a nota 2: "))
p1 = float(input("Digite o peso 1: "))
p2 = float(input("Digite a peso 2: "))

#Processamento
media = ((n1* p1) + (n2 * p2)) / (p1 + p2)
#Saída
print("Media = " + str(media))
print("Media = {:.2f}".format(media) )
