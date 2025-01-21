import pandas as pd
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_plots', type=bool, default=True)



api = wandb.Api()
entity, project = "sara_team", "prompt-decision-transformer"
runs = api.runs(entity + "/" + project)

name_list, runs_summary = [], []
for run in runs:
    if run.state == "finished" and "vel" in run.name:
        summary_list = []
        for key in run.history().keys():
            if "train-evaluation" in key:
                history_list = []
                for i, row in run.history(keys=[key]).iterrows():
                    print(int(row['_step']), key, row[key])
                    history_list.append({"step": int(row['_step']), "return": row[key]})
                summary_list.append({key: history_list})
        runs_summary.append(summary_list)
        name_list.append(run.name)
runs_df = pd.DataFrame({"summary": runs_summary, "name": name_list})
runs_df.to_csv("train_results.csv")
