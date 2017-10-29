import sys
import commands
import subprocess
import os
import shutil

def cmd(cmd):
	return commands.getoutput(cmd)

def mkdir_if_not_exists(dir):
	if not os.path.exists(dir):
	    os.mkdir(dir)

#labels
pwd = os.path.abspath(".")

dirname = sys.argv[1]
pathname = os.path.join(pwd,dirname)

labels_org = os.listdir(pathname)
labels = sorted(labels_org)
#make directries
#imageDirName = "images"
imageDirName = sys.argv[2]
imageDirNameTrain = imageDirName+"_train"
imageDirNameCV = imageDirName+"_cv"
imageDirNameTest = imageDirName+"_test"

mkdir_if_not_exists(imageDirNameTrain)
mkdir_if_not_exists(imageDirNameCV)
mkdir_if_not_exists(imageDirNameTest)

#os.mkdir(imageDirName)

#copy images and make train.txt
#imageDir = os.path.join(pwd,imageDirName)
imageDirTrain = os.path.join(pwd,imageDirNameTrain)
imageDirCV = os.path.join(pwd,imageDirNameCV)
imageDirTest = os.path.join(pwd,imageDirNameTest)

train = open('train.txt','w')
cv = open('cv.txt','w')
test = open('test.txt','w')
labelsTxt = open('labels.txt','w')

classNo=0
cnt = 0
for label in labels:
	workdir = os.path.join(pathname,label)
	if not os.path.isdir(workdir):
		continue
	images = os.listdir(workdir)
	print(label)
	labelsTxt.write(label+"\n")
	startCnt=cnt
	length = len(images)
	for image in images:
		fromimagepath = os.path.join(workdir, image)
		if not os.path.isfile(fromimagepath):
			continue
		if cnt-startCnt < length*0.6:
			imagepath = os.path.join(imageDirTrain, "image%07d" %cnt +".jpg")
			train.write(imagepath+" %d\n" % classNo)
		elif cnt-startCnt < length*0.8:
			imagepath = os.path.join(imageDirCV, "image%07d" %cnt +".jpg")
			cv.write(imagepath+" %d\n" % classNo)
		else:
			imagepath = os.path.join(imageDirTest, "image%07d" %cnt +".jpg")
			test.write(imagepath+" %d\n" % classNo)
		shutil.copyfile(fromimagepath, imagepath)
		cnt += 1

	classNo += 1

train.close()
cv.close()
test.close()
labelsTxt.close()
