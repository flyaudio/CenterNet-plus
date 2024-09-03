#coding=utf-8
# ycat			2017-08-07	  create
# 配置每个文件的 
import sys,os

if "../" not in sys.path:
	sys.path.append("../")

# if "../common" not in sys.path:
# 	sys.path.append("../common")


def setCurPath(filename):
	currentPath = os.path.dirname(filename)
	if currentPath != "":
		os.chdir(currentPath)
