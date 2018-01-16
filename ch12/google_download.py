# -*- coding: utf-8 -*-

from bs4 import BeautifulSoup
import requests
import re
#import urllib2 #python2
import urllib.request
import urllib.parse as parse #python3
import os
#import cookielib
import json

import argparse

#args = sys.argv
parser = argparse.ArgumentParser(description='This script download files from photo_kura.')
parser.add_argument('kwd', \
        action='store', \
        nargs='?', \
        const="red apple", \
        default="red apple", \
        type=str, \
        choices=None, \
        help='Keyword to search photos.', \
        metavar=None)

parser.add_argument('dir', \
        action='store', \
        nargs='?', \
        const="./Pictures_google", \
        default="./Pictures_google", \
        type=str, \
        choices=None, \
        help='Directory path where your taken photo files are located.', \
        metavar=None)

args = parser.parse_args()

def get_soup(url,header):
    return BeautifulSoup(urllib.request.urlopen(urllib.request.Request(url,headers=header)),'html.parser')
    #return BeautifulSoup(urllib2.urlopen(urllib2.Request(url,headers=header)),'html.parser')

query_org = args.kwd
label="0"
print(query_org)

#query= query.split()
#query='+'.join(query)
query = parse.quote_plus(query_org)

url="https://www.google.co.in/search?q="+query+"&source=lnms&tbm=isch"
print(url)
#add the directory for your image here
#DIR="Pictures_google"
DIR=args.dir
header={'User-Agent':"Mozilla/5.0 (Windows NT 6.1; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/43.0.2357.134 Safari/537.36"
}
soup = get_soup(url,header)

ActualImages=[]# contains the link for Large original images, type of  image
for a in soup.find_all("div",{"class":"rg_meta"}):
    link , Type =json.loads(a.text)["ou"]  ,json.loads(a.text)["ity"]
    ActualImages.append((link,Type))

print("there are total" , len(ActualImages),"images")

if not os.path.exists(DIR):
    os.mkdir(DIR)
DIR = os.path.join(DIR, query_org.split()[0])

if not os.path.exists(DIR):
    os.mkdir(DIR)
###print images
for i , (img , Type) in enumerate(ActualImages):
    try:
        req = urllib.request.Request(img,data=None,headers=header)
        #req = urllib2.Request(img, headers={'User-Agent' : header}) #python2
        response = urllib.request.urlopen(req)
        raw_img = response.read()
        #raw_img = urllib2.urlopen(req).read() #python2

        cntr = len([i for i in os.listdir(DIR) if label in i]) + 1
        print(cntr)
        if len(Type)==0:
            f = open(os.path.join(DIR , label + "_"+ str(cntr)+".jpg"), 'wb')
        else :
            f = open(os.path.join(DIR , label + "_"+ str(cntr)+"."+Type), 'wb')

        f.write(raw_img)
        f.close()
    except Exception as e:
        print("could not load : "+img)
        print(e)
