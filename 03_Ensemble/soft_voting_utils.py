
import numpy as np
import pandas as pd


ID_COL = "sample_id"

LABELS = [
    "Explicit",
    "Implicit",
    "Dodging",
    "General",
    "Deflection",
    "Declining to answer",
    "Claims ignorance",
    "Clarification",
    "Partial/half-answer",
]

PROB_COLS = [
    "p_Explicit",
    "p_Implicit",
    "p_Dodging",
    "p_General",
    "p_Deflection",
    "p_Declining_to_answer",
    "p_Claims_ignorance",
    "p_Clarification",
    "p_Partial_half_answer",
]


def load_probabilities(path):
    path = str(path)

    data = pd.read_csv(path)

    required_cols = {ID_COL, *PROB_COLS}

    if not required_cols.issubset(data.columns):
        raise ValueError(f"Missing probability columns in {path}")

    data[ID_COL] = data[ID_COL].astype(str)

    if data[ID_COL].duplicated().any():
        raise ValueError(f"Duplicate sample IDs in {path}")

    probs = data[PROB_COLS].to_numpy(dtype=float)

    if not np.isfinite(probs).all():
        raise ValueError(f"Non-finite probabilities in {path}")

    if (probs < 0).any() or (probs > 1).any():
        raise ValueError(f"Probabilities outside [0, 1] in {path}")

    if not np.allclose(probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError(f"Probability sums are invalid in {path}")

    return data[[ID_COL] + PROB_COLS]


def soft_voting(probability_frames):
    if len(probability_frames) < 2:
        raise ValueError("Soft Voting requires at least two models.")

    reference_ids = probability_frames[0][ID_COL].tolist()
    reference_set = set(reference_ids)

    probability_matrices = []

    for frame in probability_frames:
        if set(frame[ID_COL]) != reference_set:
            raise ValueError("Sample ID mismatch across models.")

        aligned = frame.set_index(ID_COL).loc[reference_ids]
        probability_matrices.append(
            aligned[PROB_COLS].to_numpy(dtype=float)
        )

    mean_probs = np.mean(
        np.stack(probability_matrices, axis=0),
        axis=0
    )

    if not np.allclose(mean_probs.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Soft Voting produced invalid probability distributions.")

    predicted_labels = np.asarray(LABELS)[
        np.argmax(mean_probs, axis=1)
    ]

    prediction_output = pd.DataFrame({
        ID_COL: reference_ids,
        "predicted_label": predicted_labels,
    })

    probability_output = pd.DataFrame(
        mean_probs,
        columns=PROB_COLS
    )

    probability_output.insert(0, ID_COL, reference_ids)

    return prediction_output, probability_output
