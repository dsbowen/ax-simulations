import os

import pandas as pd

DATA_DIR = "data"
RAW_DIR = "raw"
CLEAN_DIR = "clean"


def clean_dellavigna_pope():
    df = pd.read_stata(os.path.join(DATA_DIR, RAW_DIR, "DellaVigna_Pope.dta"))
    df = df[(df.finished == 1) & ~df.treatment.isna()]
    treatment_names = {
        1.3: "No_Payment",
        1.1: "1c_PieceRate",
        1.2: "10c_PieceRate",
        1.4: "4c_PieceRate",
        2: "Very_Low_Pay",
        3.1: "1c_RedCross",
        3.2: "10c_RedCross",
        10: "Gift_Exchange",
        4.1: "1c_2Wks",
        4.2: "1c_4Wks",
        5.1: "Gain_40c",
        5.2: "Loss_40c",
        5.3: "Gain_80c",
        6.1: "Prob.01_$1",
        6.2: "Prob.5_2c",
        7: "Social_Comp",
        8: "Ranking",
        9: "Task_Signif",
    }
    df = df.replace({"treatment": treatment_names})
    df["buttonpresses"] = df["buttonpresses"].fillna(0)
    df = df.rename(columns={"buttonpresses": "target"})
    df.to_csv(os.path.join(DATA_DIR, CLEAN_DIR, "DellaVigna_Pope.csv"), index=False)


def clean_pilot07():
    df = pd.read_csv(os.path.join(DATA_DIR, RAW_DIR, "pilot07.csv")).rename(
        columns={"treatment_index": "treatment"}
    )
    df["treatment"] = df.treatment.replace(list(range(1, 20)), "No_Payment")
    df["treatment"] = df.treatment.replace(0, "1c_PieceRate")
    df[df.completed].to_csv(
        os.path.join(DATA_DIR, CLEAN_DIR, "pilot07.csv"), index=False
    )


if __name__ == "__main__":
    # clean_dellavigna_pope()
    clean_pilot07()
