"""Gym member clustering and a targeted extras credit.

Fits k-means on a training split of gym_members.csv, checks the choice of k
on held-out members, and gives a $10 monthly credit only to segments whose
member-level net is positive. The test rows are not used to fit the scaler
or the clusters.
"""

import os
import tempfile
from pathlib import Path

# Keep matplotlib's cache writable, and tell joblib how many cores to use
# before it tries to detect physical cores.
os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "mplconfig"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "gym_members.csv"
ELBOW_PATH = ROOT / "elbow_curve.png"
DECISIONS_PATH = ROOT / "member_segments.csv"

RANDOM_SEED = 42
TEST_SIZE = 0.30
K_VALUES = range(1, 9)
N_INIT = 20
# Elbow at four clusters. Section 4 prints the training and test curves
# and stops if the test curve bends at a different k.
SELECTED_K = 4

# A member below this extras spend contributes +$5 when given the $10 credit.
# A member at or above it contributes -$10. The segment decision uses the sum.
SPEND_CUTOFF = 20.0
NET_IF_BELOW = 5.0
NET_IF_AT_OR_ABOVE = -10.0

EXPECTED_ROWS = 330
EXPECTED_COLUMNS = [
    "member_id",
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
FEATURES = [
    "age",
    "visits_per_month",
    "avg_visit_minutes",
    "classes_per_month",
    "months_member",
    "monthly_extras_spend",
]
FEATURE_LABELS = {
    "age": "age",
    "visits_per_month": "visits per month",
    "avg_visit_minutes": "minutes per visit",
    "classes_per_month": "classes per month",
    "months_member": "months as a member",
    "monthly_extras_spend": "monthly extras spend",
}

# Name rules describe the k=4 centers in original units. The gaps between
# those centers are wide, so each rule matches one segment.
SEGMENT_TRAITS = {
    "New light members": (
        "Youngest members and the newest. They visit rarely, stay for shorter "
        "sessions, almost never take a class, and spend little on extras."
    ),
    "Frequent floor members": (
        "They use the gym floor often and stay a long time, with almost no "
        "classes. Monthly extras spend is near the training average."
    ),
    "Class-focused members": (
        "They are defined by classes, about seven a month, and by the highest "
        "extras spend of the four segments."
    ),
    "Long-tenured regulars": (
        "Oldest members and the longest tenured, with a moderate visit rate. "
        "Their average extras spend is under $20. The credit rule still uses "
        "the member-level sum, not that average."
    ),
}

NEW_MEMBER = {
    "age": 30,
    "visits_per_month": 4,
    "avg_visit_minutes": 42,
    "classes_per_month": 1,
    "months_member": 5,
    "monthly_extras_spend": 10,
}


def section(title: str) -> None:
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def money(value: float) -> str:
    sign = "-" if value < 0 else ""
    return f"{sign}${abs(value):,.2f}"


def credit_net(spend: np.ndarray) -> np.ndarray:
    """Monthly net of giving the $10 credit to each member, one member at a time."""
    spend = np.asarray(spend, dtype=float)
    return np.where(spend < SPEND_CUTOFF, NET_IF_BELOW, NET_IF_AT_OR_ABOVE)


def within_cluster_spread(scaled: np.ndarray, centers: np.ndarray) -> float:
    """Sum of squared distances from each row to its nearest center."""
    distances = ((scaled[:, None, :] - centers[None, :, :]) ** 2).sum(axis=2)
    return float(distances.min(axis=1).sum())


def name_from_center(center: pd.Series) -> str:
    """Short segment name from a center in original units."""
    if center["months_member"] < 15 and center["visits_per_month"] < 6:
        return "New light members"
    if center["classes_per_month"] >= 6 and center["monthly_extras_spend"] >= 40:
        return "Class-focused members"
    if center["age"] >= 55 and center["months_member"] >= 45:
        return "Long-tenured regulars"
    if center["visits_per_month"] >= 14 and center["classes_per_month"] < 2:
        return "Frequent floor members"
    raise ValueError("No segment name matched this center: " + str(center.to_dict()))


def format_center(center: pd.Series) -> str:
    parts = []
    for feature in FEATURES:
        value = float(center[feature])
        if feature == "monthly_extras_spend":
            parts.append(f"{FEATURE_LABELS[feature]} {money(value)}")
        else:
            parts.append(f"{FEATURE_LABELS[feature]} {value:.2f}")
    return "; ".join(parts)


def make_assignment(scaler: StandardScaler, model: KMeans, segment_names: dict[int, str], credit_segments: set[str]):
    """Return a function of one member's six features.

    The returned function scales those features with the training scaler,
    assigns the nearest training center, and reports whether that segment
    receives the $10 credit.
    """

    def assign_member(
        age: float,
        visits_per_month: float,
        avg_visit_minutes: float,
        classes_per_month: float,
        months_member: float,
        monthly_extras_spend: float,
    ) -> tuple[str, bool]:
        row = pd.DataFrame(
            [
                {
                    "age": age,
                    "visits_per_month": visits_per_month,
                    "avg_visit_minutes": avg_visit_minutes,
                    "classes_per_month": classes_per_month,
                    "months_member": months_member,
                    "monthly_extras_spend": monthly_extras_spend,
                }
            ],
            columns=FEATURES,
        )
        scaled = scaler.transform(row)
        segment_id = int(model.predict(scaled)[0])
        segment = segment_names[segment_id]
        return segment, segment in credit_segments

    return assign_member


def check_members(frame: pd.DataFrame) -> None:
    print(f"File: {DATA_PATH.name}")
    print(f"Column names: {', '.join(map(str, frame.columns))}")
    missing = int(frame.isna().sum().sum())
    print(f"Rows: expected {EXPECTED_ROWS}, found {len(frame)}")
    print(f"Columns: expected {len(EXPECTED_COLUMNS)}, found {frame.shape[1]}")
    print(f"Missing values: expected 0, found {missing}")
    differences = []
    if len(frame) != EXPECTED_ROWS:
        differences.append(f"rows: expected {EXPECTED_ROWS}, found {len(frame)}")
    if frame.shape[1] != len(EXPECTED_COLUMNS):
        differences.append(
            f"columns: expected {len(EXPECTED_COLUMNS)}, found {frame.shape[1]}"
        )
    if list(frame.columns) != EXPECTED_COLUMNS:
        differences.append(
            "column names or order: expected "
            + ", ".join(EXPECTED_COLUMNS)
            + "; found "
            + ", ".join(map(str, frame.columns))
        )
    if missing != 0:
        differences.append(f"missing values: expected 0, found {missing}")
    duplicate_ids = int(frame["member_id"].duplicated().sum()) if "member_id" in frame.columns else None
    if duplicate_ids:
        differences.append(f"duplicate member_id values: {duplicate_ids}")
    if differences:
        for item in differences:
            print(f"Check failed: {item}")
        raise SystemExit("Data checks failed. gym_members.csv was not modified.")
    print("All checks passed.")
    average_spend = float(frame["monthly_extras_spend"].mean())
    print(f"Average monthly extras spend: {money(average_spend)}")
    print("Clustering uses the six descriptive features. member_id stays an identifier.")
    print("Clustering features: " + ", ".join(FEATURES))


def segment_net_table(labels: np.ndarray, names: dict[int, str], spend: np.ndarray) -> pd.DataFrame:
    rows = []
    for segment_id, name in names.items():
        member_spend = spend[labels == segment_id]
        nets = credit_net(member_spend)
        n_below = int(np.sum(member_spend < SPEND_CUTOFF))
        n_high = int(len(member_spend) - n_below)
        rows.append(
            {
                "segment_id": segment_id,
                "segment": name,
                "members": int(len(member_spend)),
                "below_cutoff": n_below,
                "at_or_above_cutoff": n_high,
                "mean_spend": float(member_spend.mean()),
                "total_net": float(nets.sum()),
            }
        )
    table = pd.DataFrame(rows)
    return table.sort_values("segment_id").reset_index(drop=True)


def format_drop(value: float | None) -> str:
    if value is None:
        return "—"
    # One decimal is enough for the large steps. A change that would round to
    # 0.0% is shown to two decimals so a small rise is still visible.
    if abs(value) < 0.0005:
        return f"{value:+.2%}"
    return f"{value:.1%}"


def print_net_table(table: pd.DataFrame) -> None:
    for row in table.itertuples(index=False):
        below_value = row.below_cutoff * NET_IF_BELOW
        high_value = row.at_or_above_cutoff * NET_IF_AT_OR_ABOVE
        direction = "positive" if row.total_net > 0 else "not positive"
        print(f"{row.segment}: {row.members} members, average extras spend {money(row.mean_spend)}")
        print(
            f"  {row.below_cutoff} below ${SPEND_CUTOFF:.0f} contribute {money(below_value)}; "
            f"{row.at_or_above_cutoff} at or above ${SPEND_CUTOFF:.0f} contribute {money(high_value)}"
        )
        print(f"  Member-level total: {money(row.total_net)} ({direction}).")


def save_elbow_chart(train_spread: list[float], test_spread: list[float], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 5))
    ks = list(K_VALUES)
    ax.plot(ks, train_spread, marker="o", linewidth=2, label="Training, model fit on these rows")
    ax.plot(ks, test_spread, marker="s", linewidth=2, label="Test, assigned to training centers")
    ax.axvline(SELECTED_K, color="0.35", linestyle="--", linewidth=1, label=f"Selected k = {SELECTED_K}")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Within-cluster spread (sum of squares)")
    ax.set_xticks(ks)
    ax.set_ylim(bottom=0)
    ax.set_title(
        "Within-cluster spread by number of clusters\n"
        "Test members use the training scaler and training centers"
    )
    ax.legend(frameon=False, loc="upper right")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def marginal_drops(spread: list[float]) -> list[float]:
    return [(previous - current) / previous for previous, current in zip(spread, spread[1:])]


def elbow_k(spread: list[float]) -> int:
    """k where the relative drop from k-1 is largest."""
    drops = marginal_drops(spread)
    return int(np.argmax(drops)) + 2


def main() -> None:
    members = pd.read_csv(DATA_PATH)

    section("1. Data check")
    check_members(members)
    members["member_id"] = members["member_id"].astype(int)

    section("2. Train/test split")
    train, test = train_test_split(members, test_size=TEST_SIZE, random_state=RANDOM_SEED)
    if len(train) + len(test) != len(members):
        raise SystemExit("Train and test splits do not cover every member.")
    if set(train.index) & set(test.index):
        raise SystemExit("Train and test splits overlap.")
    if len(test) != 99 or len(train) != 231:
        raise SystemExit(
            f"Expected 231 training rows and 99 test rows, found {len(train)} and {len(test)}."
        )
    print(f"Random seed: {RANDOM_SEED}")
    print(f"Training fraction: {1 - TEST_SIZE:.0%}. Test fraction: {TEST_SIZE:.0%}.")
    print(f"Training members: {len(train)}")
    print(f"Test members: {len(test)}")
    print("The split is shuffled with the fixed seed. It is not stratified.")
    print("The scaler and every k-means model are fit on the training rows only.")

    section("3. Standardize, then fit k = 1 through 8")
    scaler = StandardScaler()
    train_features = train[FEATURES]
    test_features = test[FEATURES]
    scaler.fit(train_features)
    x_train = scaler.transform(train_features)
    x_test = scaler.transform(test_features)
    if scaler.n_samples_seen_ != len(train):
        raise SystemExit("The scaler was not fit on the training rows alone.")
    print("StandardScaler is fit on the six training features, then applied to train and test.")
    print("member_id is not among the scaled columns.")
    print(f"Each k-means model uses k-means++ initialization, n_init={N_INIT}, random_state={RANDOM_SEED}.")

    models: dict[int, KMeans] = {}
    train_spread: list[float] = []
    test_spread: list[float] = []
    for k in K_VALUES:
        model = KMeans(n_clusters=k, init="k-means++", n_init=N_INIT, random_state=RANDOM_SEED)
        model.fit(x_train)
        if len(model.labels_) != len(train):
            raise SystemExit(f"k={k} was not fit on the training rows alone.")
        training_value = within_cluster_spread(x_train, model.cluster_centers_)
        if abs(training_value - float(model.inertia_)) > 1e-6:
            raise SystemExit(f"Training spread does not match k-means inertia at k={k}.")
        models[k] = model
        train_spread.append(training_value)
        test_spread.append(within_cluster_spread(x_test, model.cluster_centers_))

    section("4. Elbow curves and the choice of k")
    print(
        "Within-cluster spread is the sum of squared distances from each member "
        "to the center that member is assigned to, in the standardized features. "
        "For the training curve, that is the inertia of a model fit on the training rows. "
        "For the test curve, each test member is assigned to the nearest center from "
        "that same training model. The scaler and k-means are not refit on test members."
    )
    print(
        "The test total sits lower because the test split has 99 members and the "
        "training split has 231. Compare the bend in each curve, not the height."
    )
    print()
    print(
        f"{'k':>2}  {'train spread':>12}  {'train drop':>10}  "
        f"{'test spread':>11}  {'test drop':>10}  {'train / member':>14}  {'test / member':>13}"
    )
    train_drops = [None, *marginal_drops(train_spread)]
    test_drops = [None, *marginal_drops(test_spread)]
    for index, k in enumerate(K_VALUES):
        print(
            f"{k:>2}  {train_spread[index]:12.2f}  {format_drop(train_drops[index]):>10}  "
            f"{test_spread[index]:11.2f}  {format_drop(test_drops[index]):>10}  "
            f"{train_spread[index] / len(train):14.3f}  {test_spread[index] / len(test):13.3f}"
        )

    spread_at = {k: index for index, k in enumerate(K_VALUES)}
    print()
    print("Exact within-cluster spread at k=1 and at the chosen k.")
    print("Test spread uses the scaler and centers fit on the training rows only.")
    for k in (1, SELECTED_K):
        index = spread_at[k]
        print(
            f"  k={k}: training {train_spread[index]:.6f}, "
            f"test {test_spread[index]:.6f}"
        )

    training_elbow = elbow_k(train_spread)
    test_elbow = elbow_k(test_spread)
    drop_into_selected = train_drops[SELECTED_K - 1]
    drop_after_selected = train_drops[SELECTED_K]
    test_drop_into_selected = test_drops[SELECTED_K - 1]
    test_drop_after_selected = test_drops[SELECTED_K]
    print()
    later_train = max(drop for drop in train_drops[SELECTED_K:] if drop is not None)
    later_test = [drop for drop in test_drops[SELECTED_K:] if drop is not None]
    print(
        f"On the training curve, spread falls by {train_drops[1]:.1%} from k=1 to 2, "
        f"{train_drops[2]:.1%} from k=2 to 3, and {drop_into_selected:.1%} from k=3 to {SELECTED_K}. "
        f"The step into k={training_elbow} is the largest drop. "
        f"From k={SELECTED_K} to {SELECTED_K + 1} the drop is {drop_after_selected:.1%}, "
        f"and the largest later training step is {later_train:.1%}. "
        f"The bend is at k={SELECTED_K}, so v1 uses {SELECTED_K} clusters."
    )
    print(
        f"On the test curve, the largest drop is also the step into k={test_elbow} "
        f"({test_drop_into_selected:.1%} from k={SELECTED_K - 1} to {SELECTED_K}). "
        f"The steps after k={SELECTED_K} are {', '.join(format_drop(drop) for drop in later_test)}. "
        f"From k={K_VALUES[-2]} to k={K_VALUES[-1]} the test spread "
        f"{'rises' if test_spread[-1] > test_spread[-2] else 'falls'} "
        f"({test_spread[-2]:.2f} to {test_spread[-1]:.2f}). "
        f"The test curve supports k={SELECTED_K}."
    )
    if training_elbow != SELECTED_K or test_elbow != SELECTED_K:
        print(
            f"FLAG: the training elbow is k={training_elbow} and the test elbow is k={test_elbow}. "
            f"v1 is set to k={SELECTED_K}. Reconsider that choice before using the segments."
        )
        raise SystemExit("The elbow curves do not support the selected k.")
    if drop_after_selected >= 0.15 or test_drop_after_selected >= 0.15:
        print(
            "FLAG: spread is still falling by 15% or more after the selected k. "
            "Reconsider k before using the segments."
        )
        raise SystemExit("The curve has not flattened after the selected k.")
    print(
        f"No reconsideration: both curves bend at k={SELECTED_K}, "
        "and every later drop is small beside that step."
    )

    save_elbow_chart(train_spread, test_spread, ELBOW_PATH)
    print(f"Saved {ELBOW_PATH.name}")

    section("5. v1 segments, in original units")
    v1 = models[SELECTED_K]
    if v1.n_features_in_ != len(FEATURES):
        raise SystemExit("v1 was fit with a different number of features.")
    centers = pd.DataFrame(scaler.inverse_transform(v1.cluster_centers_), columns=FEATURES)
    train_labels = np.asarray(v1.labels_)
    if not np.array_equal(train_labels, v1.predict(x_train)):
        raise SystemExit("Training labels do not match assignment to the training centers.")
    segment_names = {segment_id: name_from_center(centers.loc[segment_id]) for segment_id in range(SELECTED_K)}
    if len(set(segment_names.values())) != SELECTED_K:
        raise SystemExit("Segment names are not unique.")
    missing_traits = [name for name in segment_names.values() if name not in SEGMENT_TRAITS]
    if missing_traits:
        raise SystemExit("Missing trait text for: " + ", ".join(missing_traits))

    train_means = train_features.mean()
    print("v1 is the k=4 model fit on the scaled training rows. It is not refit on the test rows.")
    print("Each center below is the training center converted back with the training scaler.")
    print("Because standardization is linear, that center is the training mean of the segment.")
    print("Training means, for comparison: " + format_center(train_means))
    print()
    ordered_ids = sorted(segment_names, key=lambda segment_id: float(centers.loc[segment_id, "months_member"]))
    ordered_names = [segment_names[segment_id] for segment_id in ordered_ids]

    def in_segment_order(table: pd.DataFrame) -> pd.DataFrame:
        return table.set_index("segment").loc[ordered_names].reset_index()

    for segment_id in ordered_ids:
        name = segment_names[segment_id]
        count = int(np.sum(train_labels == segment_id))
        print(f"{name}: {count} training members")
        print("  Center: " + format_center(centers.loc[segment_id]))
        print("  " + SEGMENT_TRAITS[name])

    section("6. Credit decision from member-level nets")
    print(
        "Giving a member the $10 credit is worth +$5 when that member's monthly "
        f"extras spend is below ${SPEND_CUTOFF:.0f}, and -$10 when it is ${SPEND_CUTOFF:.0f} or more. "
        "A segment receives the credit when the sum of these member values is positive. "
        "The decision is that member-level sum."
    )
    print("The sums below use the training members, the rows the centers were fit on.")
    print()
    train_nets = in_segment_order(
        segment_net_table(train_labels, segment_names, train["monthly_extras_spend"].to_numpy())
    )
    print_net_table(train_nets)
    credit_ids = set(train_nets.loc[train_nets["total_net"] > 0, "segment_id"].astype(int))
    credit_segments = {segment_names[segment_id] for segment_id in credit_ids}
    print()
    if not credit_segments:
        print("No training segment has a positive member-level total. v1 would give the credit to nobody.")
    else:
        print("Segments with a positive training total: " + ", ".join(sorted(credit_segments)) + ".")
    trapped = train_nets[(train_nets["mean_spend"] < SPEND_CUTOFF) & (train_nets["total_net"] <= 0)]
    if trapped.empty:
        print("No segment has average extras spend below $20 and a non-positive member-level total.")
    for row in trapped.itertuples(index=False):
        print(
            f"Average spend would misclassify {row.segment}. "
            f"Their average extras spend is {money(row.mean_spend)}, below ${SPEND_CUTOFF:.0f}, "
            f"while {row.at_or_above_cutoff} of {row.members} training members spend "
            f"${SPEND_CUTOFF:.0f} or more. The -$10 contributions outweigh the +$5 "
            f"contributions, and the segment total is {money(row.total_net)}. "
            "This segment does not receive the credit."
        )

    section("7. Assignment for one new member")
    assign_member = make_assignment(scaler, v1, segment_names, credit_segments)
    feature_text = ", ".join(
        f"{FEATURE_LABELS[feature]} {NEW_MEMBER[feature]:g}" for feature in FEATURES
    )
    print(f"New member: {feature_text}.")
    new_segment, new_credit = assign_member(
        NEW_MEMBER["age"],
        NEW_MEMBER["visits_per_month"],
        NEW_MEMBER["avg_visit_minutes"],
        NEW_MEMBER["classes_per_month"],
        NEW_MEMBER["months_member"],
        NEW_MEMBER["monthly_extras_spend"],
    )
    print(f"Assigned segment: {new_segment}")
    print(f"Receives the $10 credit: {'yes' if new_credit else 'no'}")
    new_personal_net = float(credit_net([NEW_MEMBER["monthly_extras_spend"]])[0]) if new_credit else 0.0
    print(
        "This person is outside the 330 rows in gym_members.csv. "
        f"Under the v1 rule, the monthly net of the decision is {money(new_personal_net)}."
    )

    section("8. All 330 members: baseline versus v1")
    print(
        "Every member is assigned to the nearest v1 center. "
        "The scaler and k-means stay as they were fit on the training rows."
    )
    all_labels = v1.predict(scaler.transform(members[FEATURES]))
    if len(all_labels) != len(members):
        raise SystemExit("Assignment did not cover every member.")
    print(f"Counts after assigning all {len(members)} members to the training-fitted k={SELECTED_K} centers:")
    assigned_total = 0
    for segment_id in ordered_ids:
        name = segment_names[segment_id]
        count = int(np.sum(all_labels == segment_id))
        assigned_total += count
        print(f"  {name}: {count}")
    print(f"  Total: {assigned_total}")
    if assigned_total != len(members):
        raise SystemExit("Segment counts do not add up to every member.")
    print()
    all_nets = in_segment_order(
        segment_net_table(all_labels, segment_names, members["monthly_extras_spend"].to_numpy())
    )
    all_positive = set(all_nets.loc[all_nets["total_net"] > 0, "segment"])
    print("Member-level totals if every member in the segment received the credit:")
    print()
    print_net_table(all_nets)
    print()
    if all_positive == credit_segments:
        print(
            "The same segments are positive on all 330 assigned members as on the training rows. "
            "The credit decision does not change when the held-out members are included in the sum."
        )
    else:
        print(
            "FLAG: the segments with a positive total on all 330 members are "
            + (", ".join(sorted(all_positive)) if all_positive else "(none)")
            + ". The training decision was "
            + (", ".join(sorted(credit_segments)) if credit_segments else "(none)")
            + ". Reconsider the credit segments. v1 below keeps the training decision."
        )

    receives_credit = np.array([segment_names[int(label)] in credit_segments for label in all_labels])
    spend = members["monthly_extras_spend"].to_numpy(dtype=float)
    baseline_net = credit_net(spend)
    v1_net = np.where(receives_credit, baseline_net, 0.0)
    for row in all_nets.itertuples(index=False):
        member_spend = spend[np.array([segment_names[int(label)] == row.segment for label in all_labels])]
        member_sum = float(credit_net(member_spend).sum())
        if abs(member_sum - float(row.total_net)) > 1e-6:
            raise SystemExit(f"{row.segment} total is not the sum of its member-level nets.")
        if abs(member_sum - (row.below_cutoff * NET_IF_BELOW + row.at_or_above_cutoff * NET_IF_AT_OR_ABOVE)) > 1e-6:
            raise SystemExit(f"{row.segment} total does not match the +5 and -10 member counts.")
    if abs(float(baseline_net.sum()) - float(all_nets["total_net"].sum())) > 1e-6:
        raise SystemExit("Baseline total is not the sum of the member-level segment nets.")
    print()
    print(
        "Monthly value still uses the finance assumptions one member at a time: "
        f"+{money(NET_IF_BELOW)} when that member's extras spend is below ${SPEND_CUTOFF:.0f}, "
        f"and {money(NET_IF_AT_OR_ABOVE)} when it is ${SPEND_CUTOFF:.0f} or more. "
        "Each segment total is the sum of its own members. "
        "It is not the segment's average spend."
    )
    print(
        "Baseline gives every member the credit. "
        f"Total monthly net: {money(float(baseline_net.sum()))} "
        f"on {len(members)} members."
    )
    credited = int(receives_credit.sum())
    print(
        "v1 gives the credit only to "
        + (", ".join(sorted(credit_segments)) if credit_segments else "no segment")
        + f". That is {credited} of {len(members)} members. "
        f"Total monthly net: {money(float(v1_net.sum()))}."
    )
    print(f"v1 minus baseline: {money(float(v1_net.sum() - baseline_net.sum()))}.")
    print("A positive difference means the targeted credit beats giving it to every member.")

    decisions = pd.DataFrame(
        {
            "member_id": members["member_id"].to_numpy(),
            "segment": [segment_names[int(label)] for label in all_labels],
            "receives_credit": np.where(receives_credit, "yes", "no"),
            "net_value": v1_net.astype(int),
            "monthly_extras_spend": spend,
        }
    ).sort_values("member_id")
    if len(decisions) != EXPECTED_ROWS:
        raise SystemExit("The member file does not have one row per member.")
    if set(decisions.loc[decisions["receives_credit"] == "yes", "segment"]) != credit_segments:
        raise SystemExit("Credit flags do not match the selected segments.")
    recomputed = np.where(
        decisions["receives_credit"].eq("yes"),
        credit_net(decisions["monthly_extras_spend"]),
        0.0,
    )
    if not np.array_equal(recomputed.astype(int), decisions["net_value"].to_numpy()):
        raise SystemExit("net_value does not match the member-level credit rule.")
    decisions.to_csv(DECISIONS_PATH, index=False)
    print(f"Saved {DECISIONS_PATH.name}")
    print(
        "Each row is one member. net_value is the v1 monthly net: "
        "+5 or -10 when the member's segment receives the credit, and 0 when it does not. "
        "Summing net_value reproduces the v1 total."
    )


if __name__ == "__main__":
    main()
