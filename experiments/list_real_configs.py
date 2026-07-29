import json
names=["inflation","electricity"]
names+=[d["name"] for d in json.load(open("configs/economic_manifest.json"))]
try: names+=[d["name"] for d in json.load(open("configs/eu_inflation_manifest.json"))]
except FileNotFoundError: pass
print(" ".join(names))
if __name__=="__main__": pass
