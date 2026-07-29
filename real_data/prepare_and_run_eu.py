"""Prepare, run and tabulate the EU-27 inflation panel (real stage helper)."""
import json, subprocess, sys, pathlib
def main():
    man=json.load(open("configs/eu_inflation_manifest.json"))
    for d in man:
        raw=f"data/raw/inflation_{d['geo']}.csv"; proc=f"data/processed/{d['name']}.csv"
        if not pathlib.Path(raw).exists():
            print("skip",d['geo'],"(no raw csv)"); continue
        subprocess.check_call([sys.executable,"-m","real_data.prepare_inflation","--input",raw,
                               "--output",proc,"--date-column","date","--target-column","inflation"])
        subprocess.check_call([sys.executable,"-m","experiments.run_real","--config",f"configs/{d['name']}.yaml"])
        subprocess.check_call([sys.executable,"-m","reporting.create_tables","--results-dir",f"results/{d['name']}"])
    print("EU inflation panel done.")
if __name__=="__main__": main()
