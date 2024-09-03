#coding=utf-8
import sys,os

g_currentPath = os.path.dirname(__file__)

if "../" not in sys.path:
	sys.path.append("../")

if "../common" not in sys.path:
	sys.path.append("../common")
	
def setCurPath(filename):
	currentPath = os.path.dirname(filename)
	if currentPath != "":
		os.chdir(currentPath)

