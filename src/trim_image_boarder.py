import os
import sys
import subprocess

for filename in os.listdir('.'):
	if filename.endswith('.pdf'):
		subprocess.call(['pdfcrop', filename, filename])

	if filename.endswith('.png'):
		subprocess.call(['convert', filename, '-trim', filename])
