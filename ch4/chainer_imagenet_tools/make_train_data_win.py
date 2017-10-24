import sys
import commands
import subprocess
import os
import shutil

def cmd(cmd):
	return commands.getoutput(cmd)
#	p = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
#	p.wait()
#	stdout, stderr = p.communicate()
#	return stdout.rstrip()

#labels
pwd = os.path.abspath(".")
#pathname = "./"+dirname
#print sys.argv[1]
#dirname = "101_ObjectCategories"
dirname = sys.argv[1]
pathname = os.path.join(pwd,dirname)
#dirs = cmd("dir "+dirname+" /b")
#dirs = cmd("ls "+sys.argv[1])
#labels = dirs.splitlines()
labels = os.listdir(pathname)
#make directries
imageDirName = "images"
os.mkdir(imageDirName)
#cmd("mkdir images")

#copy images and make train.txt
#pwd = cmd('pwd')

#imageDir = pwd+"/images"
#imageDirName = sys.argv[2]
#imageDir = os.path.join(pwd,imageDirName)
#imageDir = os.path.join(pwd,"images_not_cropped")
imageDir = os.path.join(pwd,imageDirName)
train = open('train.txt','w')
test = open('test.txt','w')
labelsTxt = open('labels.txt','w')

classNo=0
cnt = 0
#label = labels[classNo]
for label in labels:
	#workdir = pwd+"/"+dirname+"/"+label
	#workdir = pwd+"/"+dirname+"/"+label
	workdir = os.path.join(pathname,label)
	if not os.path.isdir(workdir):
		continue
	#imageFiles = cmd("ls "+workdir+"/*.jpg")
	#images = imageFiles.splitlines()
	images = os.listdir(workdir)
	print(label)
	labelsTxt.write(label+"\n")
	startCnt=cnt
	length = len(images)
	for image in images:
		#imagepath = imageDir+"/image%07d" %cnt +".jpg"
		imagepath = os.path.join(imageDir, "image%07d" %cnt +".jpg")
		fromimagepath = os.path.join(workdir, image)
		if not os.path.isfile(fromimagepath):
			continue
	#cmd("copy "+image+" "+imagepath)
		shutil.copyfile(fromimagepath, imagepath)
		#shutil.copyfile("C:\\src\\src.txt", "C:\\dst\\dst.txt")
		if cnt-startCnt < length*0.75:
			train.write(imagepath+" %d\n" % classNo)
		else:
			test.write(imagepath+" %d\n" % classNo)
		cnt += 1

	classNo += 1

train.close()
test.close()
labelsTxt.close()
