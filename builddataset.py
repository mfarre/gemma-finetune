import os
import json

pathrecipes = 'dump/recipes/'
files = os.listdir(pathrecipes)
recipes = []
for f in files:
    if f.endswith('.json'):
        print("Adding "+f)
        a = json.loads(open(pathrecipes+f,"r").read())
        r = ""
        for line in a['steps']:
            r = r + line + "\n"
        recipes.append({'title':a['title'],'steps':r})
b = open("recipes.json","w")
b.write(json.dumps(recipes))
b.close()