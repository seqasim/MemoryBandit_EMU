"""One-off helper to insert RPE follow-up cells into HFA_RPE_RSA_MVTemplate.ipynb."""
import json
import textwrap

NB = "/Users/salmanqasim/Documents/GitRepos/MemoryBandit_EMU/Notebooks/HFA_RPE_RSA_MVTemplate.ipynb"
INSERT_AT = 33

MD = """## Model RSA (partial RPE category) and RI category decomposition

Parsimonious follow-up: **rho_full** = cross-region RSA; **rho_item** = partial RSA controlling nominal RPE-category RDM; **ri_cat** / **ri_item** decompose trial-aligned cosine similarity **S** into category structure vs item identity beyond category. Per-subject maps are saved under `analysis_outputs/rsa_controls/model_rsa_partial/` and `ri_category_decomp/`; ACC→HPC extensive figures under `acc_hpc_reactivation_content/`.
"""

CELL_AB = r'''
# =============================
# Cells A+B: per-subject partial RSA + RI category maps (all ROI_PAIRS)
# =============================
RSA_CONTROLS_DIR = os.path.join('analysis_outputs', 'rsa_controls')
PARTIAL_OUT = os.path.join(RSA_CONTROLS_DIR, 'model_rsa_partial')
RI_CAT_OUT = os.path.join(RSA_CONTROLS_DIR, 'ri_category_decomp')
os.makedirs(PARTIAL_OUT, exist_ok=True)
os.makedirs(RI_CAT_OUT, exist_ok=True)

RUN_RPE_FOLLOWUP = True
rpe_followup_errors = []

if RUN_RPE_FOLLOWUP:
    for encoding_roi, cue_roi in ROI_PAIRS:
        pair_name = f"{encoding_roi}_to_{cue_roi}"
        partial_pair_dir = os.path.join(PARTIAL_OUT, pair_name)
        ri_pair_dir = os.path.join(RI_CAT_OUT, pair_name)
        os.makedirs(partial_pair_dir, exist_ok=True)
        os.makedirs(ri_pair_dir, exist_ok=True)
        for subj in subjects:
            try:
                s_obj = Subject(subj, base_dir, load_dir, min_trials=MIN_TRIALS, verbose=False)
                data = load_subject_crossregion_hfa(
                    subject_obj=s_obj,
                    encoding_roi=encoding_roi,
                    cue_roi=cue_roi,
                    event_encoding=EVENT_ENCODING,
                    event_cue=EVENT_CUE,
                    freq_range=FREQ_RANGE,
                    downsample_factor=DOWNSAMPLE_FACTOR,
                    win_ms=WIN_MS,
                    min_trials=MIN_TRIALS,
                )
                rpe_cat = np.asarray(data['rpe_categories'], dtype=object)
                if np.all(pd.isna(rpe_cat)):
                    raise ValueError('missing rpe_cat on combined_df for this subject')

                partial = compute_cross_temporal_rsa_partial_from_hfa(
                    data['encoding_hfa'],
                    data['cue_hfa'],
                    data['memory_outcomes'],
                    rpe_cat,
                    rdm_method=RDM_METHOD,
                    rdm_compare_method=RDM_COMPARE_METHOD,
                )
                ri_cd = compute_cross_temporal_reinstatement_category_maps(
                    data['encoding_hfa'],
                    data['cue_hfa'],
                    data['memory_outcomes'],
                    rpe_cat,
                )
                fb_t = np.asarray(data['feedback_times'], dtype=float)
                cue_t = np.asarray(data['cue_times'], dtype=float)

                np.savez_compressed(
                    os.path.join(partial_pair_dir, f'{subj}_rsa_partial_{encoding_roi}_{cue_roi}.npz'),
                    rho_full_remembered=partial['rho_full_remembered'],
                    rho_full_forgotten=partial['rho_full_forgotten'],
                    rho_full_diff=partial['rho_full_diff'],
                    rho_item_remembered=partial['rho_item_remembered'],
                    rho_item_forgotten=partial['rho_item_forgotten'],
                    rho_item_diff=partial['rho_item_diff'],
                    rho_cat_acc_remembered=partial['rho_cat_acc_remembered'],
                    rho_cat_acc_forgotten=partial['rho_cat_acc_forgotten'],
                    rho_cat_acc_diff=partial['rho_cat_acc_diff'],
                    rho_cat_cue_remembered=partial['rho_cat_cue_remembered'],
                    rho_cat_cue_forgotten=partial['rho_cat_cue_forgotten'],
                    rho_cat_cue_diff=partial['rho_cat_cue_diff'],
                    feedback_times=fb_t,
                    cue_times=cue_t,
                    rpe_categories=np.asarray(rpe_cat, dtype=object),
                    encoding_roi=np.asarray(encoding_roi),
                    cue_roi=np.asarray(cue_roi),
                    subject=np.asarray(subj),
                )
                np.savez_compressed(
                    os.path.join(ri_pair_dir, f'{subj}_ri_catdecomp_{encoding_roi}_{cue_roi}.npz'),
                    ri_item_remembered=ri_cd['ri_item_remembered'],
                    ri_item_forgotten=ri_cd['ri_item_forgotten'],
                    ri_item_diff=ri_cd['ri_item_diff'],
                    ri_cat_remembered=ri_cd['ri_cat_remembered'],
                    ri_cat_forgotten=ri_cd['ri_cat_forgotten'],
                    ri_cat_diff=ri_cd['ri_cat_diff'],
                    feedback_times=fb_t,
                    cue_times=cue_t,
                    rpe_categories=np.asarray(rpe_cat, dtype=object),
                    encoding_roi=np.asarray(encoding_roi),
                    cue_roi=np.asarray(cue_roi),
                    subject=np.asarray(subj),
                )
            except Exception as e:
                rpe_followup_errors.append((pair_name, subj, repr(e)))
    print('RPE follow-up per-subject save complete. Errors:', len(rpe_followup_errors))
    if rpe_followup_errors:
        for row in rpe_followup_errors[:20]:
            print(row)
        if len(rpe_followup_errors) > 20:
            print('...')
else:
    print('Set RUN_RPE_FOLLOWUP = True to compute partial RSA + RI category maps.')
'''

CELL_C = r'''
# =============================
# Cell C: group cluster tests + PDFs per ROI pair (rho_full/item/ri_cat/ri_item diff)
# =============================
RSA_CONTROLS_DIR = os.path.join('analysis_outputs', 'rsa_controls')
PARTIAL_OUT = os.path.join(RSA_CONTROLS_DIR, 'model_rsa_partial')
RI_CAT_OUT = os.path.join(RSA_CONTROLS_DIR, 'ri_category_decomp')
GROUP_CLUSTER_N = 5000


def _load_metric_stack_from_pair_dir(pair_dir, subjects, key):
    rows = []
    loaded = []
    fb_t = None
    cue_t = None
    for subj in subjects:
        pat = os.path.join(pair_dir, f'{subj}_rsa_partial_*.npz')
        files = glob.glob(pat)
        if len(files) != 1:
            continue
        path = files[0]
        with np.load(path, allow_pickle=True) as npz:
            if key not in npz.files:
                continue
            z = np.asarray(npz[key], dtype=np.float64)
            z = np.where(np.isfinite(z), z, 0.0)
            rows.append(z)
            loaded.append(subj)
            if fb_t is None and 'feedback_times' in npz.files:
                fb_t = np.asarray(npz['feedback_times'], dtype=float)
                cue_t = np.asarray(npz['cue_times'], dtype=float)
    if len(rows) == 0:
        return np.empty((0, 0, 0), dtype=np.float64), [], None, None
    return np.stack(rows, axis=0).astype(np.float64), loaded, fb_t, cue_t


def _load_ri_metric_stack(pair_dir, subjects, key):
    rows = []
    loaded = []
    fb_t = None
    cue_t = None
    for subj in subjects:
        pat = os.path.join(pair_dir, f'{subj}_ri_catdecomp_*.npz')
        files = glob.glob(pat)
        if len(files) != 1:
            continue
        path = files[0]
        with np.load(path, allow_pickle=True) as npz:
            if key not in npz.files:
                continue
            z = np.asarray(npz[key], dtype=np.float64)
            z = np.where(np.isfinite(z), z, 0.0)
            rows.append(z)
            loaded.append(subj)
            if fb_t is None and 'feedback_times' in npz.files:
                fb_t = np.asarray(npz['feedback_times'], dtype=float)
                cue_t = np.asarray(npz['cue_times'], dtype=float)
    if len(rows) == 0:
        return np.empty((0, 0, 0), dtype=np.float64), [], None, None
    return np.stack(rows, axis=0).astype(np.float64), loaded, fb_t, cue_t


if RUN_RPE_FOLLOWUP or os.path.isdir(PARTIAL_OUT):
    for encoding_roi, cue_roi in ROI_PAIRS:
        pair_name = f"{encoding_roi}_to_{cue_roi}"
        partial_pair_dir = os.path.join(PARTIAL_OUT, pair_name)
        ri_pair_dir = os.path.join(RI_CAT_OUT, pair_name)
        if not os.path.isdir(partial_pair_dir):
            continue

        stacks = {}
        for k in ('rho_full_diff', 'rho_item_diff'):
            stacks[k], subj_p, fb_t, cue_t = _load_metric_stack_from_pair_dir(
                partial_pair_dir, subjects, k,
            )
        for k in ('ri_cat_diff', 'ri_item_diff'):
            stacks[k], subj_r, fb2, cue2 = _load_ri_metric_stack(
                ri_pair_dir, subjects, k,
            )
            if fb_t is None:
                fb_t, cue_t = fb2, cue2

        rf = stacks.get('rho_full_diff')
        if rf is None or rf.shape[0] < 2:
            print(f'[{pair_name}] skip group cluster: need >=2 subjects with partial NPZs')
            continue

        group_bundle = {}
        for metric, D in stacks.items():
            if D is None or D.shape[0] < 2:
                continue
            group_bundle[metric] = run_group_crossregion_rsa_cluster_test(
                D,
                n_permutations=GROUP_CLUSTER_N,
                tail=1,
                threshold=0,
                seed=45 + abs(hash(metric)) % 1000,
                verbose=False,
            )

        extent = [
            float(cue_t.min()), float(cue_t.max()),
            float(fb_t.min()), float(fb_t.max()),
        ]
        for metric, D in stacks.items():
            if D is None or D.shape[0] < 2 or metric not in group_bundle:
                continue
            res = group_bundle[metric]
            gmean = np.nanmean(D, axis=0).astype(np.float32)
            sig = np.asarray(res['sig_mask'], dtype=bool)
            vmax = float(np.nanmax(np.abs(gmean)))
            if not np.isfinite(vmax) or vmax == 0:
                vmax = 1.0
            fig, ax = plt.subplots(figsize=(6, 5))
            im = ax.imshow(
                gmean,
                origin='lower',
                aspect='auto',
                extent=extent,
                cmap='RdBu_r',
                vmin=-vmax,
                vmax=vmax,
            )
            if sig.size and np.any(sig):
                ax.contour(
                    cue_t, fb_t, sig.astype(int),
                    levels=[0.5], colors='k', linewidths=1.5,
                )
            ax.axhline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.5, alpha=0.5)
            ax.set_xlabel('Cue time (s)')
            ax.set_ylabel('Feedback time (s)')
            ax.set_title(f'Group mean {metric}\n{pair_name} n={D.shape[0]}')
            fig.colorbar(im, ax=ax, shrink=0.8)
            plt.tight_layout()
            outp = os.path.join(partial_pair_dir, f'{pair_name}_{metric}_group.pdf')
            plt.savefig(outp, dpi=300, bbox_inches='tight')
            plt.show()

        np.savez_compressed(
            os.path.join(partial_pair_dir, f'{pair_name}_model_rsa_group_meta.npz'),
            rho_full_diff_stack=stacks.get('rho_full_diff', np.array([])),
            rho_item_diff_stack=stacks.get('rho_item_diff', np.array([])),
            subjects_included=np.asarray(subj_p, dtype=object),
            feedback_times=fb_t,
            cue_times=cue_t,
            encoding_roi=np.asarray(encoding_roi),
            cue_roi=np.asarray(cue_roi),
        )
        np.savez_compressed(
            os.path.join(ri_pair_dir, f'{pair_name}_ri_catdecomp_group_meta.npz'),
            ri_cat_diff_stack=stacks.get('ri_cat_diff', np.array([])),
            ri_item_diff_stack=stacks.get('ri_item_diff', np.array([])),
            subjects_included=np.asarray(subj_r, dtype=object),
            feedback_times=fb_t,
            cue_times=cue_t,
            encoding_roi=np.asarray(encoding_roi),
            cue_roi=np.asarray(cue_roi),
        )
        print(f'[{pair_name}] saved group meta + cluster PDFs under partial/ri_catdecomp dirs')
else:
    print('Skip Cell C: no PARTIAL_OUT directory')
'''

CELL_D = r'''
# =============================
# Cell D: masked scalars + simple bar (ACC -> HPC only, sig_mask_z)
# =============================
RSA_CONTROLS_DIR = os.path.join('analysis_outputs', 'rsa_controls')
PARTIAL_OUT = os.path.join(RSA_CONTROLS_DIR, 'model_rsa_partial')
RI_CAT_OUT = os.path.join(RSA_CONTROLS_DIR, 'ri_category_decomp')
MASKED_PARTIAL_CSV = os.path.join(PARTIAL_OUT, 'ACC_to_HPC', 'per_subject_masked_acc_hpc.csv')
MASKED_PARTIAL_PDF = os.path.join(PARTIAL_OUT, 'ACC_to_HPC', 'per_subject_masked_partial_acc_hpc.pdf')

acc_hpc_partial_dir = os.path.join(PARTIAL_OUT, 'ACC_to_HPC')
meta_z = os.path.join(RSA_CONTROLS_DIR, 'z_cluster', 'ACC_to_HPC_z_cluster_meta.npz')
if not os.path.isfile(meta_z):
    meta_z = os.path.join(RSA_CONTROLS_DIR, 'z_cluster', 'ACC_HPC_z_cluster_meta.npz')

if os.path.isfile(meta_z) and os.path.isdir(acc_hpc_partial_dir):
    mz = np.load(meta_z, allow_pickle=True)
    sig_mask_z_acc = np.asarray(mz['sig_mask_z'], dtype=bool)
    z_subjects_meta = [str(x) for x in np.asarray(mz['subjects_included'])]

    rows = []
    for subj in subjects:
        p_partial = os.path.join(
            acc_hpc_partial_dir, f'{subj}_rsa_partial_ACC_HPC.npz',
        )
        p_ri = os.path.join(
            RI_CAT_OUT, 'ACC_to_HPC', f'{subj}_ri_catdecomp_ACC_HPC.npz',
        )
        if not os.path.isfile(p_partial):
            continue
        with np.load(p_partial, allow_pickle=True) as npz:
            rf = np.asarray(npz['rho_full_diff'], dtype=float)
            ri = np.asarray(npz['rho_item_diff'], dtype=float)
        if os.path.isfile(p_ri):
            with np.load(p_ri, allow_pickle=True) as npz:
                ric = np.asarray(npz['ri_cat_diff'], dtype=float)
                rii = np.asarray(npz['ri_item_diff'], dtype=float)
        else:
            ric = np.full_like(rf, np.nan)
            rii = np.full_like(rf, np.nan)

        m = sig_mask_z_acc
        if m.shape != rf.shape:
            print('sig_mask_z shape mismatch; skip masked scalars')
            rows = []
            break
        if not np.any(m):
            print('empty sig_mask_z; skip masked scalars')
            rows = []
            break

        rows.append({
            'participant': subj,
            'masked_rho_full_diff': float(np.nanmean(rf[m])),
            'masked_rho_item_diff': float(np.nanmean(ri[m])),
            'masked_ri_cat_diff': float(np.nanmean(ric[m])) if np.any(np.isfinite(ric)) else np.nan,
            'masked_ri_item_diff': float(np.nanmean(rii[m])) if np.any(np.isfinite(rii)) else np.nan,
        })

    if rows:
        df_mask = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(MASKED_PARTIAL_CSV), exist_ok=True)
        df_mask.to_csv(MASKED_PARTIAL_CSV, index=False)

        x = np.arange(len(df_mask))
        fig, ax = plt.subplots(figsize=(max(8.0, 0.42 * len(df_mask)), 4.5))
        w = 0.35
        ax.bar(x - w / 2, df_mask['masked_rho_full_diff'], width=w, label='rho_full_diff', color='steelblue', alpha=0.85, edgecolor='k', linewidth=0.5)
        ax.bar(x + w / 2, df_mask['masked_rho_item_diff'], width=w, label='rho_item_diff', color='coral', alpha=0.85, edgecolor='k', linewidth=0.5)
        ax.axhline(0.0, color='k', linestyle='-', linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(df_mask['participant'], rotation=45, ha='right')
        ax.set_ylabel('mean map inside sig_mask_z')
        ax.set_title('ACC→HPC masked partial RSA (z-cluster mask)')
        ax.legend()
        plt.tight_layout()
        plt.savefig(MASKED_PARTIAL_PDF, dpi=300, bbox_inches='tight')
        plt.show()

        a = df_mask['masked_rho_full_diff'].to_numpy(dtype=float)
        b = df_mask['masked_rho_item_diff'].to_numpy(dtype=float)
        if len(a) >= 5:
            try:
                wstat, p_ab = wilcoxon(a, b, alternative='two-sided')
            except Exception:
                p_ab = np.nan
            print('Wilcoxon rho_full_diff vs rho_item_diff (paired): p =', p_ab)
        for col, lab in [
            ('masked_rho_full_diff', 'rho_full'),
            ('masked_rho_item_diff', 'rho_item'),
            ('masked_ri_cat_diff', 'ri_cat'),
            ('masked_ri_item_diff', 'ri_item'),
        ]:
            v = df_mask[col].dropna().to_numpy(dtype=float)
            if len(v) >= 5:
                try:
                    w0, p0 = wilcoxon(v, alternative='two-sided')
                except Exception:
                    p0 = np.nan
                print(f'Wilcoxon {lab} vs 0: p =', p0)
        print('Saved', MASKED_PARTIAL_CSV, MASKED_PARTIAL_PDF)
else:
    print('Skip Cell D: need z_cluster meta + ACC_to_HPC partial NPZs')
'''

# Cell E is very long — loaded from companion file if present
try:
    from pathlib import Path
    cell_e_path = Path(__file__).resolve().parent / '_rpe_followup_cell_e.py'
    if cell_e_path.is_file():
        CELL_E = cell_e_path.read_text()
    else:
        CELL_E = "# Cell E: run _rpe_followup_cell_e.py content or see repository\npass\n"
except Exception:
    CELL_E = "pass\n"


def _src(s: str):
    return [line + "\n" for line in textwrap.dedent(s).strip("\n").split("\n")]


def main():
    with open(NB, encoding="utf-8") as f:
        nb = json.load(f)
    new_cells = [
        {"cell_type": "markdown", "metadata": {}, "source": _src(MD)},
        {"cell_type": "code", "metadata": {}, "source": _src(CELL_AB)},
        {"cell_type": "code", "metadata": {}, "source": _src(CELL_C)},
        {"cell_type": "code", "metadata": {}, "source": _src(CELL_D)},
        {"cell_type": "code", "metadata": {}, "source": _src(CELL_E)},
    ]
    nb["cells"] = nb["cells"][:INSERT_AT] + new_cells + nb["cells"][INSERT_AT:]
    with open(NB, "w", encoding="utf-8") as f:
        json.dump(nb, f, indent=1, ensure_ascii=False)
    print("Inserted", len(new_cells), "cells at", INSERT_AT)


if __name__ == "__main__":
    main()
