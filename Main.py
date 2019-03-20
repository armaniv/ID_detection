#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# istogramma -> La parte sinistra dell'asse orizzontale rappresenta le aree scure e nere, la parte in mezzo le aree
# grigie e la parte destra le aree bianche e più chiare. L'asse verticale rappresenta la grandezza dell'area che è stata
# catturata in ognuna di queste zone. Così l'istogramma di un'immagine molto luminosa con poche aree scure e/o ombre
# avrà più punti verso la destra e il centro del grafico, al contrario per un'immagine poco luminosa.

import os

ROOTDIR = '/home/vale/Documenti/Project Course'

for subdir, dirs, files in os.walk(ROOTDIR):
    print (os.path.join(subdir))

