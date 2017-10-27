import sys
import commands
import subprocess
import os
import shutil

def cmd(cmd):
	return commands.getoutput(cmd)

#labels
pwd = os.path.abspath(".")

dirname = sys.argv[1]
pathname = os.path.join(pwd,dirname)

labels = os.listdir(pathname)
#make directries
#imageDirName = "images"
imageDirName = sys.argv[2]
os.mkdir(imageDirName)

#copy images and make train.txt
imageDir = os.path.join(pwd,imageDirName)
train = open('train.txt','w')
train = open('cv.txt','w')
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
		imagepath = os.path.join(imageDir, "image%07d" %cnt +".jpg")
		fromimagepath = os.path.join(workdir, image)
		if not os.path.isfile(fromimagepath):
			continue
		shutil.copyfile(fromimagepath, imagepath)
		if cnt-startCnt < length*0.6:
			train.write(imagepath+" %d\n" % classNo)
		elif: cnt-startCnt < length*0.8:
			cv.write(imagepath+" %d\n" % classNo)
		else:
			test.write(imagepath+" %d\n" % classNo)
		cnt += 1

	classNo += 1

train.close()
cv.close()
test.close()
labelsTxt.close()
