import pandas as pd
import wandb
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--test_plots', type=bool, default=True)


def get_csv_results():
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


import numpy as np


def fix_results():
    api = wandb.Api()
    entity, project = "sara_team", "prompt-decision-transformer"
    runs = api.runs(f"{entity}/{project}")

    for run in runs:
        if run.state == "finished" and "feature-True" in run.name and "J-2-H-20" in run.name and "fix" not in run.name:

            print(run.name)
            wandb.init(entity=entity, project=project, name=run.name + "_fixed", reinit=True)
            log_data = {}
            for key in run.history().keys():
                if "return_mean" in key:
                    print(key)
                    history = run.history(keys=[key])
                    df = pd.DataFrame(history)
                    df["_step"] = range(250)
                    numerical_columns = list(df.select_dtypes(include=["number"]).columns)
                    log_data.update({numerical_columns[1]: list(df[numerical_columns[1]])})

            for i in range(250):
                logs = {k:v[i] for k, v in log_data.items()}
                # print(logs)
                wandb.log(logs)
            wandb.finish()


if __name__ == '__main__':
    fix_results()
