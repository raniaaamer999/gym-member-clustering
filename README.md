# Gym member clustering

This project clusters gym members with k-means, checks that choice of k on held-out members, and gives a $10 monthly extras credit only to segments that have a positive member-level net. `gym_members.csv` is the input and is not modified.

## Setup

From this folder, use the existing environment:

```bash
source .venv/bin/activate
python analysis.py
```

To recreate the environment:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python analysis.py
```

On macOS or Linux, activate the environment with `source .venv/bin/activate`. On Windows, use `.venv\Scripts\activate`.

The script uses pandas, scikit-learn, and matplotlib.

## What the script does

- Checks that the CSV has 330 rows, 7 columns, and no missing values, then prints the average monthly extras spend. `member_id` is kept as an identifier and is not a clustering feature.
- Holds out 30% of the rows with random seed 42 (231 training members, 99 test members).
- Fits `StandardScaler` on the six training features only, then scales the training and test rows with that scaler.
- Fits k-means for k = 1 through 8 on the scaled training rows, with random seed 42 and `n_init=20`. Test members are assigned to those training centers. The scaler and k-means are not refit on the test rows.
- Saves `elbow_curve.png`, with training within-cluster spread and test spread on the same figure. Spread is the sum of squared distances to the assigned center. On this file both curves bend at k = 4 and the later drops are much smaller, so v1 uses four clusters.
- Describes each v1 segment from its training center, converted back to the original units, with a short name, the training count, and the defining traits.
- Decides the credit from the training members, one member at a time: +$5 when monthly extras spend is below $20, and −$10 when it is $20 or more. A segment receives the credit only when that sum is positive. One segment averages under $20 and still has a negative sum, so it does not receive the credit.
- Assigns a new member (age 30, 4 visits per month, 42 minutes per visit, 1 class per month, 5 months as a member, $10 extras) with a function of those six features.
- Assigns all 330 members to the training-fitted v1 centers and prints the total monthly net of giving every member the credit versus giving it only to the selected segments.
- Saves `member_segments.csv` with each member's segment, credit decision, and v1 net.

Running `python analysis.py` prints the checks, the elbow table, the segment centers, the credit arithmetic, the new member's assignment, and both plan totals. Those printed numbers are the results to cite.
