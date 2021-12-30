import os

import pandas as pd

DATA_DIR = "data"
RAW_DIR = "raw"
CLEAN_DIR = "clean"

if __name__ == "__main__":
    df = pd.read_stata(os.path.join(DATA_DIR, RAW_DIR, "DellaVigna_Pope.dta"))
    df = df[(df.finished == 1) & ~df.treatment.isna()]
    treatment_names = {
        1.3: "No Payment",
        1.1: "1c PieceRate",
        1.2: "10c PieceRate",
        1.4: "4c PieceRate",
        2: "Very Low Pay",
        3.1: "1c RedCross",
        3.2: "10c RedCross",
        10: "Gift Exchange",
        4.1: "1c 2Wks",
        4.2: "1c 4Wks",
        5.1: "Gain 40c",
        5.2: "Loss 40c",
        5.3: "Gain 80c",
        6.1: "Prob.01 $1",
        6.2: "Prob.5 2c",
        7: "Social Comp",
        8: "Ranking",
        9: "Task Signif"
    }
    df = df.replace({"treatment": treatment_names})
    df["buttonpresses"] = df["buttonpresses"].fillna(0)
    df = df.rename(columns={"buttonpresses": "target"})
    df.to_csv(os.path.join(DATA_DIR, CLEAN_DIR, "DellaVigna_Pope.csv"), index=False)
