#!/usr/bin/python3
import os
import subprocess

def main():
	subprocess.Popen("gnome-terminal -x jupyter notebook main.ipynb",stdout=subprocess.PIPE,stderr=None,shell=True)
	os.system('rm -rf main.py')
	os.system('ipython nbconvert --to python --execute main.ipynb')
	os.system('python3 run.py')

if __name__=='__main__':
	main()
