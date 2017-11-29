#! /usr/bin/python
# -*- coding: utf-8 -*-

#try:
#    import urllib.request as urllib2
#except ImportError:
#    import urllib2

import urllib.request
from urllib.error import HTTPError
import urllib.parse
import json, sys
import unicodedata


def translate(phrase):
    #try:
    #    phrase=phrase.decode('utf-8')
    #except UnicodeDecodeError:
    #    phrase=phrase.decode('cp932')

    if is_japanese(phrase):
        from_lang = u"ja"# English
        dest_lang = u"en"# Japanese
    else:
        from_lang = u"en"# English
        dest_lang = u"ja"# Japanese



    url_param = {}
    url_param["from"]=from_lang
    url_param["dest"]=dest_lang
    url_param["format"]="json"
    url_param["phrase"]=phrase
    url_param["pretty"]="true"

    urldata = urllib.parse.urlencode(url_param)
    url = "https://glosbe.com/gapi/translate/?" + urldata
    #print(url)
    #urldata = urldata.encode('ascii')
    #req = urllib.request.Request(url, urldata)
    #response = urllib.request.urlopen(url)
    try:
        with urllib.request.urlopen(url) as response:
            json_data = response.read().decode("utf-8")
            json_dict = json.loads(json_data)
    except HTTPError as e:
        content = e.read()
        print("ERROR")
        print(content)
    #url = u"https://glosbe.com/gapi/translate?from=" \
    #    + from_lang + u"&dest=" + dest_lang \
    #    + u"&format=json&phrase=" + phrase + u"&pretty=true"
    #print(url)

    return_txt = ""
    tuc = json_dict["tuc"]# tuc: list
    for i in range(len(tuc)):
        if u"phrase" in tuc[i].keys():
            return_txt += tuc[i]["phrase"]["text"] + ","
    return return_txt

def is_japanese(string):
    for ch in string:
        name = unicodedata.name(ch)
        if "CJK UNIFIED" in name \
        or "HIRAGANA" in name \
        or "KATAKANA" in name:
            return True
    return False

if __name__ == '__main__':
    argvs = sys.argv
    argc = len(argvs)

    if argc == 2:
        phrase=sys.argv[1]

    else:
        # 使い方を教える
        print("Usage: python translate.py 'word'")
        phrase = ""
    if phrase:
        phrase = translate(phrase)
        if phrase:
            print(phrase)
        else:
            print("Not Found. ")
