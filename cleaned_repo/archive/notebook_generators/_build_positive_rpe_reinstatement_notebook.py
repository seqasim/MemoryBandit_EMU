from pathlib import Path
from textwrap import dedent

import nbformat as nbf


NOTEBOOK_PATH = Path(
    "/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/"
    "MemoryBandit-PositiveRPE_Reinstatement_SingleSubject.ipynb"
)


def md(text: str):
    return nbf.v4.new_markdown_cell(dedent(text).strip() + "\n")


def code(text: str):
    return nbf.v4.new_code_cell(dedent(text).strip() + "\n")


cells = [
    md(
        """
        # ACC Category-Congruent Reactivation and Memory

        This notebook asks a simple question in plain language:

        **Are positive-RPE memories helped more by reactivating their matching ACC state than negative- or neutral-RPE memories are helped by reactivating theirs?**

        The analysis is intentionally presentation-first. It is designed to:

        - explain each step in non-technical language,
        - print readable summaries as it goes,
        - and produce figures that can stand on their own in slides or reports.

        ## Main idea

        During feedback, each subject's ACC activity is used to learn three neural templates:

        - a **negative-RPE** ACC state,
        - a **neutral-RPE** ACC state,
        - and a **positive-RPE** ACC state.

        During recognition, we ask whether the cue-period ACC pattern looks more like the state that originally belonged to that item.

        For example:

        - a positive-RPE item is scored by how strongly its cue-period ACC activity matches the **positive** template relative to the other two,
        - a negative-RPE item is scored by how strongly it matches the **negative** template relative to the other two,
        - and a neutral-RPE item is scored the same way using the **neutral** template.

        ## Why this notebook focuses on old items

        Old/studied items have a real encoding history, so they have a real encoding RPE category.  
        Lures do not have that history, so they do **not** have a category-congruent reinstatement score in the same sense.

        For that reason, the primary analysis here includes only:

        - **hits**
        - **misses**

        Lures are intentionally left out of this notebook revision.
        """
    ),
    code(
        """
        %matplotlib inline
        %reload_ext autoreload
        %autoreload 2

        import os
        import warnings
        from pathlib import Path

        os.environ.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")

        import matplotlib.pyplot as plt
        import numpy as np
        import pandas as pd
        import seaborn as sns
        from IPython.display import Markdown, display
        from joblib import Parallel, delayed
        from sklearn.linear_model import LogisticRegression, Ridge

        from plot_config import set_plot_style
        from mv_regression_utils import Subject

        warnings.filterwarnings("ignore")
        set_plot_style()
        sns.set_style("whitegrid")

        base_dir = "/Volumes/T7 Shield"
        load_dir = f"{base_dir}/work/qasims01/MemoryBanditData/EMU"
        demo_file = f"{base_dir}/projects/guLab/Salman/EMU/MemoryBanditsDatabase.xlsx"

        ROI = "ACC"
        EVENT_ENCODING = "feedback_start"
        EVENT_PROJECTION = "cue_on"
        FREQ_RANGE = (70.0, 201.0)
        DOWNSAMPLE_FACTOR = 5
        WIN_MS = 200
        MIN_TRIALS = 10
        MIN_TRIALS_PER_CLASS = 3
        RIDGE_ALPHA = 1.0
        N_JOBS = -1

        FEEDBACK_WINDOW = (0.83, 1.50)
        CUE_WINDOW = (0.37, 0.60)

        RPE_ORDER = ["negative", "neutral", "positive"]
        DISPLAY_ORDER = ["positive", "negative", "neutral"]
        OUTCOME_ORDER = ["miss", "hit"]
        CATEGORY_COLORS = {
            "positive": "#2ca02c",
            "negative": "#d62728",
            "neutral": "#7f7f7f",
        }
        IOWA_EXCLUDE = ["MS015", "MS024", "MS034", "MS038"]

        HFA_SMOOTH_METHOD = "movmean"
        HFA_ARTIFACT_ABS_THRESH = 10.0

        demo_data = pd.read_excel(demo_file, engine="openpyxl")
        subj_df = demo_data[(demo_data.LongVersion == 1) & (demo_data.Minerva == 1)]
        subj_df = subj_df[~subj_df["MSSMCode"].isin(IOWA_EXCLUDE)]
        SUBJECTS = [str(x) for x in subj_df.MSSMCode.values]

        print("Notebook configuration")
        print("----------------------")
        print(f"ROI: {ROI}")
        print(f"Feedback training window: {FEEDBACK_WINDOW[0]:.2f} to {FEEDBACK_WINDOW[1]:.2f} s")
        print(f"Cue projection window:    {CUE_WINDOW[0]:.2f} to {CUE_WINDOW[1]:.2f} s")
        print(f"Frequency range:          {FREQ_RANGE[0]:.1f} to {FREQ_RANGE[1]:.1f} Hz")
        print(f"Candidate subjects:       {len(SUBJECTS)}")
        """
    ),
    md(
        """
        ## Analysis setup

        The analysis uses two fixed windows that were chosen from prior ACC results:

        - **feedback / template training:** `0.83–1.50 s`
        - **cue / retrieval projection:** `0.37–0.60 s`

        In simple terms:

        1. learn what the subject's ACC looks like during negative, neutral, and positive RPE states,
        2. measure how much that ACC pattern reappears during recognition,
        3. ask whether that "matched-state reactivation" is more strongly linked to memory for positive-RPE items.
        """
    ),
    code(
        """
        def print_header(title):
            line = "=" * len(title)
            print(f"\\n{title}\\n{line}")


        def normalize_rpe_category(value):
            if pd.isna(value):
                return np.nan
            text = str(value).strip().lower()
            mapping = {
                "neg": "negative",
                "negative": "negative",
                "neu": "neutral",
                "neutral": "neutral",
                "pos": "positive",
                "positive": "positive",
            }
            return mapping.get(text, np.nan)


        def safe_zscore(values):
            arr = pd.Series(values, copy=False)
            std = arr.std(ddof=0)
            if pd.isna(std) or std < 1e-12:
                return pd.Series(np.zeros(len(arr), dtype=float), index=arr.index)
            return (arr - arr.mean()) / std


        def pick_window_mask(times, window):
            times = np.asarray(times, dtype=float)
            mask = (times >= float(window[0])) & (times <= float(window[1]))
            return mask


        def summarize_counts(df, label):
            print_header(label)
            if len(df) == 0:
                print("No rows available.")
                return
            print(df.to_string(index=False))


        def compute_congruent_reactivation(row):
            if row["rpe_category"] == "positive":
                return row["E_pos_ctr"] - 0.5 * (row["E_neg_ctr"] + row["E_neu_ctr"])
            if row["rpe_category"] == "negative":
                return row["E_neg_ctr"] - 0.5 * (row["E_pos_ctr"] + row["E_neu_ctr"])
            if row["rpe_category"] == "neutral":
                return row["E_neu_ctr"] - 0.5 * (row["E_pos_ctr"] + row["E_neg_ctr"])
            return np.nan


        def mean_ci(values):
            vals = np.asarray(values, dtype=float)
            vals = vals[np.isfinite(vals)]
            if vals.size == 0:
                return np.nan, np.nan, np.nan
            mean = float(np.mean(vals))
            if vals.size == 1:
                return mean, np.nan, np.nan
            sem = float(np.std(vals, ddof=1) / np.sqrt(vals.size))
            ci = 1.96 * sem
            return mean, mean - ci, mean + ci


        def add_caption(ax, text):
            ax.text(
                0.0,
                -0.25,
                text,
                transform=ax.transAxes,
                fontsize=10,
                ha="left",
                va="top",
            )


        def describe_series(series, label):
            vals = pd.Series(series).dropna()
            print_header(label)
            if len(vals) == 0:
                print("No valid values.")
                return
            print(vals.describe().to_string())


        def fit_subject_logistic_slope(sub_df):
            x = np.asarray(sub_df["congruent_reactivation_z"], dtype=float)
            y = np.asarray(sub_df["hit"], dtype=int)
            ok = np.isfinite(x) & np.isfinite(y)
            x = x[ok]
            y = y[ok]
            if x.size < 4 or np.unique(y).size < 2:
                return np.nan
            model = LogisticRegression(penalty=None, solver="lbfgs", max_iter=2000)
            model.fit(x.reshape(-1, 1), y)
            return float(model.coef_[0, 0])
        """
    ),
    md(
        """
        ## Subject-level neural processing

        For each subject, the code below:

        - loads ACC high-frequency activity during feedback and recognition,
        - averages each trial within the fixed training and retrieval windows,
        - trains three subject-specific ACC templates from feedback trials,
        - projects recognition trials onto those templates,
        - and keeps only old-item trials that can be linked back to a real encoding RPE category.

        The result is a trialwise table that can be used for statistics, summaries, and figures.
        """
    ),
    code(
        """
        def process_subject(subject_id):
            out = {
                "subject": subject_id,
                "ok": False,
                "error": "",
                "n_feedback_trials": np.nan,
                "n_cue_trials": np.nan,
                "n_old_trials": 0,
                "n_old_hits": 0,
                "n_old_misses": 0,
                "class_counts": {},
                "old_rows": [],
            }

            try:
                subj = Subject(subject_id, base_dir, load_dir, min_trials=MIN_TRIALS, verbose=False)

                neural_fb = subj._get_or_build_event_neural_data(
                    event=EVENT_ENCODING,
                    roi=ROI,
                    freq_range=FREQ_RANGE,
                    downsample_factor=DOWNSAMPLE_FACTOR,
                    win_ms=WIN_MS,
                    smooth_method=HFA_SMOOTH_METHOD,
                    artifact_abs_thresh=HFA_ARTIFACT_ABS_THRESH,
                )
                neural_cue = subj._get_or_build_event_neural_data(
                    event=EVENT_PROJECTION,
                    roi=ROI,
                    freq_range=FREQ_RANGE,
                    downsample_factor=DOWNSAMPLE_FACTOR,
                    win_ms=WIN_MS,
                    smooth_method=HFA_SMOOTH_METHOD,
                    artifact_abs_thresh=HFA_ARTIFACT_ABS_THRESH,
                )

                fb_hfa = np.asarray(neural_fb["hfa"], dtype=np.float32)
                cue_hfa = np.asarray(neural_cue["hfa"], dtype=np.float32)
                fb_times = np.asarray(neural_fb["times"], dtype=np.float32)
                cue_times = np.asarray(neural_cue["times"], dtype=np.float32)
                fb_labels = np.asarray(neural_fb["electrode_labels"], dtype=str)
                cue_labels = np.asarray(neural_cue["electrode_labels"], dtype=str)

                common_elec = [lab for lab in fb_labels if lab in set(cue_labels)]
                if len(common_elec) == 0:
                    out["error"] = "no_common_electrodes"
                    return out

                fb_ix = np.asarray([np.where(fb_labels == lab)[0][0] for lab in common_elec], dtype=int)
                cue_ix = np.asarray([np.where(cue_labels == lab)[0][0] for lab in common_elec], dtype=int)
                fb_hfa = fb_hfa[:, fb_ix, :]
                cue_hfa = cue_hfa[:, cue_ix, :]

                fb_mask = pick_window_mask(fb_times, FEEDBACK_WINDOW)
                cue_mask = pick_window_mask(cue_times, CUE_WINDOW)
                if int(fb_mask.sum()) == 0:
                    out["error"] = "empty_feedback_window"
                    return out
                if int(cue_mask.sum()) == 0:
                    out["error"] = "empty_cue_window"
                    return out

                fb_pattern = np.nanmean(fb_hfa[:, :, fb_mask], axis=2)
                cue_pattern = np.nanmean(cue_hfa[:, :, cue_mask], axis=2)
                n_trials_fb = fb_pattern.shape[0]
                n_trials_cue = cue_pattern.shape[0]
                out["n_feedback_trials"] = int(n_trials_fb)
                out["n_cue_trials"] = int(n_trials_cue)

                learn_df = subj.learn_df.loc[subj.learn_df.participant == subject_id].copy()
                learn_df["rpe_category"] = learn_df["rpe_cat"].map(normalize_rpe_category)
                learn_df["trial_ix"] = pd.to_numeric(learn_df["trials"], errors="coerce") - 1
                learn_df = learn_df.dropna(subset=["trial_ix", "rpe_category"]).copy()
                learn_df["trial_ix"] = learn_df["trial_ix"].astype(int)
                learn_df = learn_df[
                    (learn_df["trial_ix"] >= 0)
                    & (learn_df["trial_ix"] < n_trials_fb)
                    & (learn_df["rpe_category"].isin(RPE_ORDER))
                ].copy()
                learn_df = learn_df.sort_values("trial_ix").drop_duplicates("trial_ix", keep="last")
                if len(learn_df) < MIN_TRIALS:
                    out["error"] = f"too_few_training_trials_{len(learn_df)}"
                    return out

                class_counts = learn_df["rpe_category"].value_counts().reindex(RPE_ORDER, fill_value=0)
                out["class_counts"] = class_counts.to_dict()
                if int((class_counts < MIN_TRIALS_PER_CLASS).sum()) > 0:
                    out["error"] = "insufficient_trials_per_rpe_class"
                    return out

                X_train = np.asarray(fb_pattern[learn_df["trial_ix"].to_numpy()], dtype=np.float64)
                row_ok = np.all(np.isfinite(X_train), axis=1)
                X_train = X_train[row_ok]
                y_cat = learn_df.loc[row_ok, "rpe_category"].reset_index(drop=True)
                if X_train.shape[0] < MIN_TRIALS:
                    out["error"] = "too_few_finite_training_rows"
                    return out

                x_mu = np.nanmean(X_train, axis=0)
                x_sd = np.nanstd(X_train, axis=0, ddof=0)
                x_sd = np.where(np.isfinite(x_sd) & (x_sd > 1e-8), x_sd, 1.0)
                X_train_z = (X_train - x_mu) / x_sd
                X_cue_z = (np.asarray(cue_pattern, dtype=np.float64) - x_mu) / x_sd

                y_df = pd.get_dummies(pd.Categorical(y_cat, categories=RPE_ORDER), dtype=float)
                model = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True)
                model.fit(X_train_z, y_df.to_numpy(dtype=float))

                cue_evidence = np.asarray(model.predict(X_cue_z), dtype=np.float64)
                trial_to_cat = learn_df.set_index("trial_ix")["rpe_category"].to_dict()
                trial_to_rpe = learn_df.set_index("trial_ix")["rpe"].to_dict()

                cdf = subj.combined_df[
                    (subj.combined_df.participant == subject_id)
                    & (subj.combined_df.condition == "Day1")
                ].copy()
                if "trials_mem" not in cdf.columns:
                    out["error"] = "missing_trials_mem_column"
                    return out

                cdf["trials_mem"] = pd.to_numeric(cdf["trials_mem"], errors="coerce")
                cdf["trials_gamble"] = pd.to_numeric(cdf["trials_gamble"], errors="coerce")
                cdf = cdf.dropna(subset=["trials_mem"]).copy()
                cdf["cue_trial_ix"] = cdf["trials_mem"].astype(int) - 1
                cdf["encoding_trial_ix"] = cdf["trials_gamble"] - 1
                cdf = cdf[
                    (cdf["cue_trial_ix"] >= 0)
                    & (cdf["cue_trial_ix"] < n_trials_cue)
                ].copy()
                cdf["is_old_trial"] = cdf["hits"].astype(bool) | cdf["misses"].astype(bool)
                old_cdf = cdf.loc[cdf["is_old_trial"]].copy()

                rows = []
                for _, row in old_cdf.iterrows():
                    encoding_ix = row["encoding_trial_ix"]
                    if pd.isna(encoding_ix):
                        continue
                    encoding_ix = int(encoding_ix)
                    rpe_category = trial_to_cat.get(encoding_ix, np.nan)
                    if rpe_category not in RPE_ORDER:
                        continue
                    cue_ix_single = int(row["cue_trial_ix"])
                    E_neg, E_neu, E_pos = cue_evidence[cue_ix_single, :]
                    rows.append(
                        {
                            "subject": subject_id,
                            "cue_trial_ix": cue_ix_single,
                            "encoding_trial_ix": encoding_ix,
                            "trial_id": f"{subject_id}_mem{int(row['trials_mem'])}",
                            "hit": int(bool(row["hits"])),
                            "memory_outcome": "hit" if bool(row["hits"]) else "miss",
                            "recog_rt": float(row["recog_rt"]) if pd.notna(row.get("recog_rt", np.nan)) else np.nan,
                            "rpe_category": rpe_category,
                            "rpe_value": float(trial_to_rpe.get(encoding_ix, row.get("rpe", np.nan))),
                            "E_neg": float(E_neg),
                            "E_neu": float(E_neu),
                            "E_pos": float(E_pos),
                        }
                    )

                out["old_rows"] = rows
                out["n_old_trials"] = len(rows)
                out["n_old_hits"] = int(sum(r["hit"] for r in rows))
                out["n_old_misses"] = int(sum(1 - r["hit"] for r in rows))
                out["ok"] = len(rows) > 0
                if not out["ok"]:
                    out["error"] = "no_old_trials_after_alignment"
                return out
            except Exception as exc:
                out["error"] = f"{type(exc).__name__}: {exc}"
                return out
        """
    ),
    code(
        """
        print_header("Running subject-level ACC template and reactivation pipeline")

        results = Parallel(n_jobs=N_JOBS, backend="loky")(
            delayed(process_subject)(sid) for sid in SUBJECTS
        )

        ok_res = [r for r in results if r.get("ok")]
        fail_df = pd.DataFrame(
            [{"subject": r["subject"], "error": r.get("error", "")} for r in results if not r.get("ok")]
        )

        print(f"Successful subjects: {len(ok_res)} / {len(results)}")
        if len(fail_df):
            print("\\nSubjects excluded or not processed successfully:")
            display(fail_df)

        subject_summary = pd.DataFrame(
            [
                {
                    "subject": r["subject"],
                    "feedback_trials": r["n_feedback_trials"],
                    "cue_trials": r["n_cue_trials"],
                    "old_trials": r["n_old_trials"],
                    "old_hits": r["n_old_hits"],
                    "old_misses": r["n_old_misses"],
                    "n_negative_train": r["class_counts"].get("negative", np.nan),
                    "n_neutral_train": r["class_counts"].get("neutral", np.nan),
                    "n_positive_train": r["class_counts"].get("positive", np.nan),
                }
                for r in ok_res
            ]
        ).sort_values("subject")

        summarize_counts(subject_summary, "Subject-level usable data summary")

        all_rows = []
        for r in ok_res:
            all_rows.extend(r["old_rows"])

        old_df = pd.DataFrame(all_rows)
        print(f"\\nTotal aligned old-item trials: {len(old_df)}")
        """
    ),
    md(
        """
        ## Build the main old-item analysis table

        Each row below represents one old/studied recognition trial.

        The table keeps:

        - whether the item was a **hit** or a **miss**,
        - the item's true encoding RPE category,
        - the cue-period evidence for each ACC RPE state,
        - and the final **category-congruent reactivation** score used in the main model.
        """
    ),
    code(
        """
        if len(old_df) == 0:
            raise RuntimeError("No aligned old-item trials were available for analysis.")

        old_df["subject"] = old_df["subject"].astype(str)
        old_df["rpe_category"] = pd.Categorical(old_df["rpe_category"], categories=DISPLAY_ORDER, ordered=True)
        old_df["memory_outcome"] = pd.Categorical(old_df["memory_outcome"], categories=OUTCOME_ORDER, ordered=True)

        for col in ["E_neg", "E_neu", "E_pos"]:
            old_df[f"{col}_z"] = old_df.groupby("subject")[col].transform(safe_zscore)

        old_df["E_mean_z"] = old_df[["E_neg_z", "E_neu_z", "E_pos_z"]].mean(axis=1)
        old_df["E_neg_ctr"] = old_df["E_neg_z"] - old_df["E_mean_z"]
        old_df["E_neu_ctr"] = old_df["E_neu_z"] - old_df["E_mean_z"]
        old_df["E_pos_ctr"] = old_df["E_pos_z"] - old_df["E_mean_z"]

        old_df["congruent_reactivation"] = old_df.apply(compute_congruent_reactivation, axis=1)
        old_df["congruent_reactivation_z"] = old_df.groupby("subject")["congruent_reactivation"].transform(safe_zscore)
        old_df["log_recog_rt"] = np.where(
            np.isfinite(old_df["recog_rt"]) & (old_df["recog_rt"] > 0),
            np.log(old_df["recog_rt"]),
            np.nan,
        )

        print_header("Main old-item dataframe")
        print(f"Rows: {len(old_df)}")
        print(f"Subjects contributing at least one old item: {old_df['subject'].nunique()}")

        outcome_counts = (
            old_df.groupby(["rpe_category", "memory_outcome"], observed=False)
            .size()
            .rename("n_trials")
            .reset_index()
        )
        summarize_counts(outcome_counts, "Old-item trial counts by RPE category and memory outcome")

        for label in ["E_neg", "E_neu", "E_pos", "congruent_reactivation", "congruent_reactivation_z"]:
            describe_series(old_df[label], f"Descriptive summary: {label}")
        describe_series(old_df["recog_rt"], "Descriptive summary: recog_rt")
        describe_series(old_df["log_recog_rt"], "Descriptive summary: log_recog_rt")

        display(old_df.head())
        """
    ),
    md(
        """
        ## Readable descriptive summaries and report-ready figures

        The next cells summarize the data in a way that is easy to report:

        - how many trials contributed to each condition,
        - how category-matched reactivation is distributed,
        - whether hits tend to show stronger reactivation than misses,
        - and whether that pattern is strongest for positive-RPE items.
        """
    ),
    code(
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 4.8))

        sns.countplot(
            data=old_df,
            x="rpe_category",
            hue="memory_outcome",
            order=DISPLAY_ORDER,
            hue_order=OUTCOME_ORDER,
            palette={"hit": "#2ca02c", "miss": "#d62728"},
            ax=axes[0],
        )
        axes[0].set_title("Old-item sample composition")
        axes[0].set_xlabel("Encoding RPE category")
        axes[0].set_ylabel("Number of retrieval trials")
        axes[0].legend(title="Recognition outcome")
        add_caption(
            axes[0],
            "Each bar shows how many old/studied retrieval trials contributed to the main analysis.",
        )

        sns.violinplot(
            data=old_df,
            x="rpe_category",
            y="congruent_reactivation_z",
            order=DISPLAY_ORDER,
            palette=CATEGORY_COLORS,
            inner=None,
            cut=0,
            ax=axes[1],
        )
        sns.stripplot(
            data=old_df.sample(min(len(old_df), 800), random_state=42),
            x="rpe_category",
            y="congruent_reactivation_z",
            order=DISPLAY_ORDER,
            color="black",
            alpha=0.25,
            size=3,
            ax=axes[1],
        )
        axes[1].axhline(0, color="black", lw=0.8, alpha=0.6)
        axes[1].set_title("Distribution of category-matched ACC reactivation")
        axes[1].set_xlabel("Encoding RPE category")
        axes[1].set_ylabel("Within-subject z-scored congruent reactivation")
        add_caption(
            axes[1],
            "Higher values mean stronger expression of the ACC state that matches the item's original RPE category.",
        )

        sns.pointplot(
            data=old_df,
            x="rpe_category",
            y="congruent_reactivation_z",
            hue="memory_outcome",
            order=DISPLAY_ORDER,
            hue_order=OUTCOME_ORDER,
            dodge=0.25,
            errorbar=95,
            palette={"hit": "#2ca02c", "miss": "#d62728"},
            ax=axes[2],
        )
        axes[2].axhline(0, color="black", lw=0.8, alpha=0.6)
        axes[2].set_title("Hits vs misses within each RPE category")
        axes[2].set_xlabel("Encoding RPE category")
        axes[2].set_ylabel("Mean congruent reactivation (z)")
        axes[2].legend(title="Recognition outcome")
        add_caption(
            axes[2],
            "If hits sit above misses, memory success is associated with stronger category-matched ACC reactivation.",
        )

        fig.suptitle("Descriptive overview of the old-item dataset", y=1.04, fontsize=16)
        fig.tight_layout()
        plt.show()
        """
    ),
    code(
        """
        subj_effect_rows = []
        for (subject, category), sub_df in old_df.groupby(["subject", "rpe_category"], observed=False):
            if len(sub_df) == 0:
                continue
            hit_vals = sub_df.loc[sub_df["hit"] == 1, "congruent_reactivation_z"]
            miss_vals = sub_df.loc[sub_df["hit"] == 0, "congruent_reactivation_z"]
            subj_effect_rows.append(
                {
                    "subject": subject,
                    "rpe_category": category,
                    "n_trials": len(sub_df),
                    "n_hits": int((sub_df["hit"] == 1).sum()),
                    "n_misses": int((sub_df["hit"] == 0).sum()),
                    "hit_minus_miss": float(hit_vals.mean() - miss_vals.mean()) if len(hit_vals) and len(miss_vals) else np.nan,
                    "logistic_slope": fit_subject_logistic_slope(sub_df),
                }
            )

        subj_effect_df = pd.DataFrame(subj_effect_rows)
        summarize_counts(
            subj_effect_df.sort_values(["rpe_category", "subject"]),
            "Subject-level within-category memory association summary",
        )

        fig, ax = plt.subplots(figsize=(8, 5))
        sns.stripplot(
            data=subj_effect_df,
            x="rpe_category",
            y="hit_minus_miss",
            order=DISPLAY_ORDER,
            palette=CATEGORY_COLORS,
            size=7,
            alpha=0.8,
            ax=ax,
        )
        sns.pointplot(
            data=subj_effect_df,
            x="rpe_category",
            y="hit_minus_miss",
            order=DISPLAY_ORDER,
            errorbar=95,
            color="black",
            markers="D",
            linestyles="none",
            ax=ax,
        )
        ax.axhline(0, color="black", lw=0.8, alpha=0.7)
        ax.set_title("Subject-level memory benefit from category-matched ACC reactivation")
        ax.set_xlabel("Encoding RPE category")
        ax.set_ylabel("Hit minus miss congruent reactivation (z)")
        add_caption(
            ax,
            "Each point is one subject. Values above zero mean hits had stronger category-matched ACC reactivation than misses.",
        )
        plt.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Main model

        The primary statistical question is:

        **Is the relationship between category-matched ACC reactivation and memory stronger for positive-RPE items than for negative- or neutral-RPE items?**

        The intended long-term model is a Bayesian mixed logistic model:

        `hit ~ congruent_reactivation_z * rpe_category + (1|subject)`

        For now, this notebook uses a **temporary linear mixed-effects approximation**:

        - `hit` is treated as a continuous `0/1` variable as a shorthand,
        - the model uses a random intercept for subject,
        - and the goal is a stable, readable approximation while the preferred logistic/Bayesian version is deferred.
        """
    ),
    code(
        """
        import statsmodels.api as sm
        import statsmodels.formula.api as smf
        from scipy.stats import norm

        old_model_df = old_df.dropna(
            subset=["hit", "subject", "rpe_category", "congruent_reactivation_z"]
        ).copy()
        old_model_df["rpe_category"] = pd.Categorical(
            old_model_df["rpe_category"],
            categories=DISPLAY_ORDER,
            ordered=True,
        )

        print_header("Model input summary")
        print(f"Rows entering the model: {len(old_model_df)}")
        print(f"Subjects entering the model: {old_model_df['subject'].nunique()}")
        print(
            old_model_df.groupby(["rpe_category", "memory_outcome"], observed=False)
            .size()
            .rename("n_trials")
            .reset_index()
            .to_string(index=False)
        )

        MIXEDLM_FORMULA = (
            "hit ~ congruent_reactivation_z * "
            "C(rpe_category, Treatment('positive'))"
        )

        print_header("Temporary linear mixed-effects model")
        print(MIXEDLM_FORMULA)
        mixedlm_model = smf.mixedlm(
            MIXEDLM_FORMULA,
            data=old_model_df,
            groups=old_model_df["subject"],
            re_formula="1",
        )
        mixedlm_result = mixedlm_model.fit(reml=False, method="lbfgs", maxiter=2000)

        print(f"Converged: {getattr(mixedlm_result, 'converged', 'unknown')}")
        print(f"Log-likelihood: {mixedlm_result.llf:.3f}")
        print(f"Random intercept variance: {float(np.asarray(mixedlm_result.cov_re).ravel()[0]):.6f}")
        print("\\nFixed effects summary:")
        print(mixedlm_result.summary())
        """
    ),
    md(
        """
        ### Commented Bambi model for later

        The cell below keeps the preferred Bayesian/logistic model visible for later restoration, but it is intentionally commented out and does not run in this version of the notebook.
        """
    ),
    code(
        """
        # BAMBI_FORMULA = "hit ~ congruent_reactivation_z * rpe_category + (1|subject)"
        #
        # import bambi as bmb
        # import arviz as az
        #
        # bambi_model = bmb.Model(BAMBI_FORMULA, old_model_df, family="bernoulli")
        # bambi_idata = bambi_model.fit(
        #     draws=1000,
        #     tune=1000,
        #     chains=4,
        #     cores=4,
        #     target_accept=0.95,
        # )
        # display(az.summary(bambi_idata, var_names=None, round_to=3))
        """
    ),
    code(
        """
        def linear_contrast(params, cov, contrast):
            est = float(contrast @ params)
            se = float(np.sqrt(max(contrast @ cov @ contrast, 0.0)))
            z = est / se if se > 0 else np.nan
            p = 2 * (1 - norm.cdf(abs(z))) if np.isfinite(z) else np.nan
            ci_low = est - 1.96 * se if np.isfinite(se) else np.nan
            ci_high = est + 1.96 * se if np.isfinite(se) else np.nan
            return {
                "estimate": est,
                "se": se,
                "z": z,
                "p": p,
                "ci_low": ci_low,
                "ci_high": ci_high,
            }

        fe_params = mixedlm_result.fe_params.copy()
        fe_names = list(fe_params.index)
        fe_cov = mixedlm_result.cov_params().loc[fe_names, fe_names].to_numpy(dtype=float)
        fe_beta = fe_params.to_numpy(dtype=float)

        coef_index = {name: idx for idx, name in enumerate(fe_names)}

        def make_contrast(weights):
            vec = np.zeros(len(fe_names), dtype=float)
            for name, value in weights.items():
                vec[coef_index[name]] = value
            return vec

        base_term = "congruent_reactivation_z"
        neg_int = "congruent_reactivation_z:C(rpe_category, Treatment('positive'))[T.negative]"
        neu_int = "congruent_reactivation_z:C(rpe_category, Treatment('positive'))[T.neutral]"

        slope_contrasts = {
            "positive": make_contrast({base_term: 1.0}),
            "negative": make_contrast({base_term: 1.0, neg_int: 1.0}),
            "neutral": make_contrast({base_term: 1.0, neu_int: 1.0}),
        }

        slope_rows = []
        for category, contrast in slope_contrasts.items():
            row = linear_contrast(fe_beta, fe_cov, contrast)
            row["category"] = category
            slope_rows.append(row)
        mixedlm_slope_df = pd.DataFrame(slope_rows)[
            ["category", "estimate", "se", "z", "p", "ci_low", "ci_high"]
        ]
        summarize_counts(mixedlm_slope_df, "MixedLM category-specific memory slopes")

        primary_contrasts = {
            "positive - negative": slope_contrasts["positive"] - slope_contrasts["negative"],
            "positive - neutral": slope_contrasts["positive"] - slope_contrasts["neutral"],
            "positive - mean(negative, neutral)": (
                slope_contrasts["positive"] - 0.5 * (slope_contrasts["negative"] + slope_contrasts["neutral"])
            ),
        }

        contrast_rows = []
        for label, contrast in primary_contrasts.items():
            row = linear_contrast(fe_beta, fe_cov, contrast)
            row["contrast"] = label
            contrast_rows.append(row)
        mixedlm_contrast_df = pd.DataFrame(contrast_rows)[
            ["contrast", "estimate", "se", "z", "p", "ci_low", "ci_high"]
        ]
        summarize_counts(
            mixedlm_contrast_df,
            "MixedLM primary slope contrasts",
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        slope_plot_df = mixedlm_slope_df.copy()
        slope_plot_df["category"] = pd.Categorical(
            slope_plot_df["category"], categories=DISPLAY_ORDER, ordered=True
        )
        slope_plot_df = slope_plot_df.sort_values("category")

        axes[0].errorbar(
            x=np.arange(len(slope_plot_df)),
            y=slope_plot_df["estimate"],
            yerr=[
                slope_plot_df["estimate"] - slope_plot_df["ci_low"],
                slope_plot_df["ci_high"] - slope_plot_df["estimate"],
            ],
            fmt="o",
            color="black",
            ecolor="black",
            capsize=4,
        )
        for idx, row in enumerate(slope_plot_df.itertuples(index=False)):
            axes[0].scatter(idx, row.estimate, s=80, color=CATEGORY_COLORS[row.category], zorder=3)
        axes[0].axhline(0, color="black", lw=0.8, alpha=0.7)
        axes[0].set_xticks(np.arange(len(slope_plot_df)))
        axes[0].set_xticklabels(slope_plot_df["category"])
        axes[0].set_title("Linear mixed-model slope within each RPE category")
        axes[0].set_xlabel("Encoding RPE category")
        axes[0].set_ylabel("Estimated linear slope for congruent reactivation")
        add_caption(
            axes[0],
            "Higher slopes mean memory is more strongly associated with category-matched ACC reactivation.",
        )

        contrast_plot_df = mixedlm_contrast_df.copy()
        axes[1].errorbar(
            x=np.arange(len(contrast_plot_df)),
            y=contrast_plot_df["estimate"],
            yerr=[
                contrast_plot_df["estimate"] - contrast_plot_df["ci_low"],
                contrast_plot_df["ci_high"] - contrast_plot_df["estimate"],
            ],
            fmt="o",
            color="black",
            ecolor="black",
            capsize=4,
        )
        axes[1].axhline(0, color="black", lw=0.8, alpha=0.7)
        axes[1].set_xticks(np.arange(len(contrast_plot_df)))
        axes[1].set_xticklabels(contrast_plot_df["contrast"], rotation=18, ha="right")
        axes[1].set_title("Linear mixed-model primary slope contrasts")
        axes[1].set_xlabel("Contrast")
        axes[1].set_ylabel("Difference in linear slope")
        add_caption(
            axes[1],
            "Positive values mean the positive-RPE memory slope exceeds the comparison slope.",
        )

        fig.suptitle("Main model results", y=1.03, fontsize=16)
        fig.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Parallel model: log recognition RT

        The next section repeats the same mixed-effects structure, but now asks whether category-matched ACC reactivation predicts **recognition response time** rather than hit/miss outcome.

        Here the dependent variable is:

        - `log_recog_rt = log(recog_rt)`

        and the analysis stays restricted to the same aligned old/studied trials used above.
        """
    ),
    code(
        """
        rt_model_df = old_df.dropna(
            subset=["log_recog_rt", "subject", "rpe_category", "congruent_reactivation_z"]
        ).copy()
        rt_model_df["rpe_category"] = pd.Categorical(
            rt_model_df["rpe_category"],
            categories=DISPLAY_ORDER,
            ordered=True,
        )

        print_header("RT model input summary")
        print(f"Rows entering the RT model: {len(rt_model_df)}")
        print(f"Subjects entering the RT model: {rt_model_df['subject'].nunique()}")
        print(
            rt_model_df.groupby(["rpe_category", "memory_outcome"], observed=False)
            .size()
            .rename("n_trials")
            .reset_index()
            .to_string(index=False)
        )

        RT_MIXEDLM_FORMULA = (
            "log_recog_rt ~ congruent_reactivation_z * "
            "C(rpe_category, Treatment('positive'))"
        )

        print_header("Linear mixed-effects model for log recognition RT")
        print(RT_MIXEDLM_FORMULA)
        mixedlm_rt_model = smf.mixedlm(
            RT_MIXEDLM_FORMULA,
            data=rt_model_df,
            groups=rt_model_df["subject"],
            re_formula="1",
        )
        mixedlm_rt_result = mixedlm_rt_model.fit(reml=False, method="lbfgs", maxiter=2000)

        print(f"Converged: {getattr(mixedlm_rt_result, 'converged', 'unknown')}")
        print(f"Log-likelihood: {mixedlm_rt_result.llf:.3f}")
        print(f"Random intercept variance: {float(np.asarray(mixedlm_rt_result.cov_re).ravel()[0]):.6f}")
        print("\\nFixed effects summary:")
        print(mixedlm_rt_result.summary())
        """
    ),
    code(
        """
        rt_fe_params = mixedlm_rt_result.fe_params.copy()
        rt_fe_names = list(rt_fe_params.index)
        rt_fe_cov = mixedlm_rt_result.cov_params().loc[rt_fe_names, rt_fe_names].to_numpy(dtype=float)
        rt_fe_beta = rt_fe_params.to_numpy(dtype=float)

        rt_coef_index = {name: idx for idx, name in enumerate(rt_fe_names)}

        def make_rt_contrast(weights):
            vec = np.zeros(len(rt_fe_names), dtype=float)
            for name, value in weights.items():
                vec[rt_coef_index[name]] = value
            return vec

        rt_base_term = "congruent_reactivation_z"
        rt_neg_int = "congruent_reactivation_z:C(rpe_category, Treatment('positive'))[T.negative]"
        rt_neu_int = "congruent_reactivation_z:C(rpe_category, Treatment('positive'))[T.neutral]"

        rt_slope_contrasts = {
            "positive": make_rt_contrast({rt_base_term: 1.0}),
            "negative": make_rt_contrast({rt_base_term: 1.0, rt_neg_int: 1.0}),
            "neutral": make_rt_contrast({rt_base_term: 1.0, rt_neu_int: 1.0}),
        }

        rt_slope_rows = []
        for category, contrast in rt_slope_contrasts.items():
            row = linear_contrast(rt_fe_beta, rt_fe_cov, contrast)
            row["category"] = category
            rt_slope_rows.append(row)
        mixedlm_rt_slope_df = pd.DataFrame(rt_slope_rows)[
            ["category", "estimate", "se", "z", "p", "ci_low", "ci_high"]
        ]
        summarize_counts(mixedlm_rt_slope_df, "MixedLM category-specific RT slopes")

        rt_primary_contrasts = {
            "positive - negative": rt_slope_contrasts["positive"] - rt_slope_contrasts["negative"],
            "positive - neutral": rt_slope_contrasts["positive"] - rt_slope_contrasts["neutral"],
            "positive - mean(negative, neutral)": (
                rt_slope_contrasts["positive"] - 0.5 * (rt_slope_contrasts["negative"] + rt_slope_contrasts["neutral"])
            ),
        }

        rt_contrast_rows = []
        for label, contrast in rt_primary_contrasts.items():
            row = linear_contrast(rt_fe_beta, rt_fe_cov, contrast)
            row["contrast"] = label
            rt_contrast_rows.append(row)
        mixedlm_rt_contrast_df = pd.DataFrame(rt_contrast_rows)[
            ["contrast", "estimate", "se", "z", "p", "ci_low", "ci_high"]
        ]
        summarize_counts(
            mixedlm_rt_contrast_df,
            "MixedLM primary RT contrasts",
        )

        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        rt_slope_plot_df = mixedlm_rt_slope_df.copy()
        rt_slope_plot_df["category"] = pd.Categorical(
            rt_slope_plot_df["category"], categories=DISPLAY_ORDER, ordered=True
        )
        rt_slope_plot_df = rt_slope_plot_df.sort_values("category")

        axes[0].errorbar(
            x=np.arange(len(rt_slope_plot_df)),
            y=rt_slope_plot_df["estimate"],
            yerr=[
                rt_slope_plot_df["estimate"] - rt_slope_plot_df["ci_low"],
                rt_slope_plot_df["ci_high"] - rt_slope_plot_df["estimate"],
            ],
            fmt="o",
            color="black",
            ecolor="black",
            capsize=4,
        )
        for idx, row in enumerate(rt_slope_plot_df.itertuples(index=False)):
            axes[0].scatter(idx, row.estimate, s=80, color=CATEGORY_COLORS[row.category], zorder=3)
        axes[0].axhline(0, color="black", lw=0.8, alpha=0.7)
        axes[0].set_xticks(np.arange(len(rt_slope_plot_df)))
        axes[0].set_xticklabels(rt_slope_plot_df["category"])
        axes[0].set_title("Linear mixed-model RT slope within each RPE category")
        axes[0].set_xlabel("Encoding RPE category")
        axes[0].set_ylabel("Estimated slope for log recognition RT")
        add_caption(
            axes[0],
            "More negative slopes mean stronger category-matched reactivation is associated with faster recognition responses.",
        )

        rt_contrast_plot_df = mixedlm_rt_contrast_df.copy()
        axes[1].errorbar(
            x=np.arange(len(rt_contrast_plot_df)),
            y=rt_contrast_plot_df["estimate"],
            yerr=[
                rt_contrast_plot_df["estimate"] - rt_contrast_plot_df["ci_low"],
                rt_contrast_plot_df["ci_high"] - rt_contrast_plot_df["estimate"],
            ],
            fmt="o",
            color="black",
            ecolor="black",
            capsize=4,
        )
        axes[1].axhline(0, color="black", lw=0.8, alpha=0.7)
        axes[1].set_xticks(np.arange(len(rt_contrast_plot_df)))
        axes[1].set_xticklabels(rt_contrast_plot_df["contrast"], rotation=18, ha="right")
        axes[1].set_title("Linear mixed-model RT contrasts")
        axes[1].set_xlabel("Contrast")
        axes[1].set_ylabel("Difference in RT slope")
        add_caption(
            axes[1],
            "These contrasts test whether the reactivation-RT relationship is more negative or more positive for positive-RPE items than for the comparison categories.",
        )

        fig.suptitle("Log recognition RT model results", y=1.03, fontsize=16)
        fig.tight_layout()
        plt.show()
        """
    ),
    md(
        """
        ## Plain-language takeaway

        This final cell prints a short reporting-oriented summary that can be pasted into notes or slides.
        """
    ),
    code(
        """
        print_header("Plain-language reporting summary")

        n_subjects = old_model_df["subject"].nunique()
        n_trials = len(old_model_df)
        print(
            f"This analysis included {n_trials} old-item retrieval trials from {n_subjects} subjects."
        )
        print(
            "Each old item was scored by how strongly cue-period ACC activity matched the item's own encoding RPE state."
        )
        print(
            "The key comparison was whether this reactivation-memory relationship was larger for positive-RPE items than for the other RPE categories."
        )
        print(
            "This version treats hit as a continuous 0/1 variable in a temporary linear mixed-effects shorthand."
        )

        print("\\nLinear mixed-model slope estimates by category:")
        print(mixedlm_slope_df.to_string(index=False))

        print("\\nLinear mixed-model primary contrasts:")
        print(mixedlm_contrast_df.to_string(index=False))

        print("\\nLog recognition RT slope estimates by category:")
        print(mixedlm_rt_slope_df.to_string(index=False))

        print("\\nLog recognition RT primary contrasts:")
        print(mixedlm_rt_contrast_df.to_string(index=False))
        """
    ),
]


nb = nbf.v4.new_notebook()
nb["cells"] = cells
nb["metadata"] = {
    "kernelspec": {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    },
    "language_info": {
        "name": "python",
        "version": "3.x",
    },
}

NOTEBOOK_PATH.write_text(nbf.writes(nb))
print(f"Wrote notebook to {NOTEBOOK_PATH}")
