import subprocess
import sys
import os

for figures in os.listdir('.'):
	if figures.endswith('pdf'):
		print(figures)
		subprocess.call(['pdfcrop', figures, figures])
		pass
