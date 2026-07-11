# =============================
# Cell E: ACC→HPC extensive plot pack (figures 1–8 + combined PDF)
# =============================
from scipy.spatial.distance import squareform

try:
    RSA_CONTROLS_DIR
except NameError:
    RSA_CONTROLS_DIR = os.path.join('analysis_outputs', 'rsa_controls')
try:
    PARTIAL_OUT
except NameError:
    PARTIAL_OUT = os.path.join(RSA_CONTROLS_DIR, 'model_rsa_partial')
try:
    RI_CAT_OUT
except NameError:
    RI_CAT_OUT = os.path.join(RSA_CONTROLS_DIR, 'ri_category_decomp')

ACC_HPC_PLOT_DIR = os.path.join(RSA_CONTROLS_DIR, 'acc_hpc_reactivation_content')
os.makedirs(ACC_HPC_PLOT_DIR, exist_ok=True)

ENC_ROI, CUE_ROI = 'ACC', 'HPC'
pair_tag = f'{ENC_ROI}_to_{CUE_ROI}'
partial_dir = os.path.join(PARTIAL_OUT, pair_tag)
ri_dir = os.path.join(RI_CAT_OUT, pair_tag)
meta_z_path = os.path.join(RSA_CONTROLS_DIR, 'z_cluster', 'ACC_to_HPC_z_cluster_meta.npz')
if not os.path.isfile(meta_z_path):
    meta_z_path = os.path.join(RSA_CONTROLS_DIR, 'z_cluster', 'ACC_HPC_z_cluster_meta.npz')


def _stack_npz(pair_dir, subjects, fname_fmt, key):
    """fname_fmt uses {subj}, e.g. '{subj}_rsa_partial_ACC_HPC.npz'."""
    rows = []
    loaded = []
    fb_t = cue_t = None
    for subj in subjects:
        path = os.path.join(pair_dir, fname_fmt.format(subj=subj))
        if not os.path.isfile(path):
            continue
        with np.load(path, allow_pickle=True) as npz:
            if key not in npz.files:
                continue
            a = np.asarray(npz[key], dtype=np.float64)
            a = np.where(np.isfinite(a), a, np.nan)
            rows.append(a)
            loaded.append(subj)
            if fb_t is None:
                fb_t = np.asarray(npz['feedback_times'], dtype=float)
                cue_t = np.asarray(npz['cue_times'], dtype=float)
    if not rows:
        return None, [], None, None
    return np.stack(rows, axis=0), loaded, fb_t, cue_t


if not (os.path.isdir(partial_dir) and os.path.isdir(ri_dir) and os.path.isfile(meta_z_path)):
    print('Cell E skipped: need partial_dir, ri_dir, and z_cluster meta')
else:
    mz = np.load(meta_z_path, allow_pickle=True)
    sig_mask_z = np.asarray(mz['sig_mask_z'], dtype=bool)

    fmt_p = '{subj}_rsa_partial_ACC_HPC.npz'
    fmt_r = '{subj}_ri_catdecomp_ACC_HPC.npz'

    stacks = {}
    for k in (
        'rho_full_remembered', 'rho_full_forgotten', 'rho_full_diff',
        'rho_item_remembered', 'rho_item_forgotten', 'rho_item_diff',
        'rho_cat_acc_remembered', 'rho_cat_acc_forgotten', 'rho_cat_acc_diff',
        'rho_cat_cue_remembered', 'rho_cat_cue_forgotten', 'rho_cat_cue_diff',
    ):
        stacks[k], loaded_p, fb_t, cue_t = _stack_npz(partial_dir, subjects, fmt_p, k)
    for k in (
        'ri_cat_remembered', 'ri_cat_forgotten', 'ri_cat_diff',
        'ri_item_remembered', 'ri_item_forgotten', 'ri_item_diff',
    ):
        stacks[k], loaded_r, fb2, cue2 = _stack_npz(ri_dir, subjects, fmt_r, k)
        if fb_t is None and fb2 is not None:
            fb_t, cue_t = fb2, cue2

    if stacks.get('rho_full_diff') is None or stacks['rho_full_diff'].shape[0] < 2:
        print('Cell E: insufficient stacks')
    else:
        extent = [
            float(cue_t.min()), float(cue_t.max()),
            float(fb_t.min()), float(fb_t.max()),
        ]  # imshow extent: x = cue, y = feedback

        diff_names = ('rho_full_diff', 'rho_item_diff', 'ri_cat_diff', 'ri_item_diff')
        cluster_info = {}
        for name in diff_names:
            D = stacks[name]
            if D is None or D.shape[0] < 2:
                cluster_info[name] = None
                continue
            cluster_info[name] = run_group_crossregion_rsa_cluster_test(
                D, n_permutations=5000, tail=1, threshold=0,
                seed=200 + hash(name) % 800, verbose=False,
            )

        def _vmax99_list(arrs):
            parts = [a.ravel() for a in arrs if a is not None]
            if not parts:
                return 1.0
            v = np.abs(np.concatenate(parts))
            v = v[np.isfinite(v)]
            return float(np.percentile(v, 99)) if v.size else 1.0

        # ----- Figure 1: 4 x 3
        row_specs = [
            ('rho_full', 'rho_full_remembered', 'rho_full_forgotten', 'rho_full_diff'),
            ('rho_item', 'rho_item_remembered', 'rho_item_forgotten', 'rho_item_diff'),
            ('ri_cat', 'ri_cat_remembered', 'ri_cat_forgotten', 'ri_cat_diff'),
            ('ri_item', 'ri_item_remembered', 'ri_item_forgotten', 'ri_item_diff'),
        ]
        all_m = []
        for _, kr, kf, kd in row_specs:
            for kk in (kr, kf, kd):
                if stacks.get(kk) is not None:
                    all_m.append(np.nanmean(stacks[kk], axis=0))
        vmax1 = max(_vmax99_list(all_m), 1e-6)

        fig1, axes1 = plt.subplots(4, 3, figsize=(14, 16))
        for ri, (label, kr, kf, kd) in enumerate(row_specs):
            kd_cluster = cluster_info.get(kd)
            sig_k = None
            if kd_cluster is not None and kd_cluster.get('sig_mask') is not None:
                sig_k = np.asarray(kd_cluster['sig_mask'], dtype=bool)
            for ci, (k, title_suffix) in enumerate([(kr, 'Hits'), (kf, 'Misses'), (kd, 'Hits − Misses')]):
                ax = axes1[ri, ci]
                gmean = np.nanmean(stacks[k], axis=0)
                ax.imshow(
                    gmean, origin='lower', aspect='auto', extent=extent,
                    cmap='RdBu_r', vmin=-vmax1, vmax=vmax1,
                )
                if title_suffix == 'Hits − Misses' and sig_k is not None and sig_k.shape == gmean.shape and np.any(sig_k):
                    ax.contour(
                        cue_t, fb_t, sig_k.astype(int),
                        levels=[0.5], colors='k', linewidths=1.2,
                    )
                ax.axhline(0, color='k', linestyle='--', linewidth=0.4, alpha=0.5)
                ax.axvline(0, color='k', linestyle='--', linewidth=0.4, alpha=0.5)
                ax.set_title(f'{label} | {title_suffix}', fontsize=9)
                if ci == 0:
                    ax.set_ylabel(f'Feedback ({ENC_ROI})')
                if ri == 3:
                    ax.set_xlabel(f'Cue ({CUE_ROI})')
        fig1.suptitle(
            f'ACC→HPC | n={stacks["rho_full_diff"].shape[0]} | diff column: cluster p<=0.05',
            fontsize=11, y=1.01,
        )
        plt.tight_layout()
        fig1.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig1_acc_hpc_group_grid.pdf'), dpi=300, bbox_inches='tight')
        fig1.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig1_acc_hpc_group_grid.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # ----- Figure 2: category encoding 2 x 3
        cat_acc_keys = (
            'rho_cat_acc_remembered', 'rho_cat_acc_forgotten', 'rho_cat_acc_diff',
        )
        cat_cue_keys = (
            'rho_cat_cue_remembered', 'rho_cat_cue_forgotten', 'rho_cat_cue_diff',
        )
        all_cat = [np.nanmean(stacks[k], axis=0) for k in cat_acc_keys + cat_cue_keys if stacks.get(k) is not None]
        vmax2 = max(_vmax99_list(all_cat), 1e-6)

        fig2, axes2 = plt.subplots(2, 3, figsize=(14, 7))
        for ci, k in enumerate(cat_acc_keys):
            gmean = np.nanmean(stacks[k], axis=0)
            axes2[0, ci].imshow(
                gmean, origin='lower', aspect='auto', extent=extent,
                cmap='RdBu_r', vmin=-vmax2, vmax=vmax2,
            )
            axes2[0, ci].set_title(f'ACC vs M_rpe | {k.split("_")[-1]}', fontsize=9)
        for ci, k in enumerate(cat_cue_keys):
            gmean = np.nanmean(stacks[k], axis=0)
            axes2[1, ci].imshow(
                gmean, origin='lower', aspect='auto', extent=extent,
                cmap='RdBu_r', vmin=-vmax2, vmax=vmax2,
            )
            axes2[1, ci].set_title(f'HPC vs M_rpe | {k.split("_")[-1]}', fontsize=9)
        for ax in axes2.ravel():
            ax.axhline(0, color='k', linestyle='--', linewidth=0.4, alpha=0.5)
            ax.axvline(0, color='k', linestyle='--', linewidth=0.4, alpha=0.5)
        plt.tight_layout()
        fig2.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig2_acc_hpc_category_encoding.pdf'), dpi=300, bbox_inches='tight')
        fig2.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig2_acc_hpc_category_encoding.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # ----- Figures 3–4, 8: masked CSV from Cell D
        dfm_path = os.path.join(partial_dir, 'per_subject_masked_acc_hpc.csv')
        if os.path.isfile(dfm_path):
            dfm = pd.read_csv(dfm_path)
            a = dfm['masked_rho_full_diff'].to_numpy(dtype=float)
            b = dfm['masked_rho_item_diff'].to_numpy(dtype=float)

            fig3, axes3 = plt.subplots(1, 2, figsize=(12, 5))
            xx = np.arange(len(dfm))
            axes3[0].bar(xx - 0.2, a, 0.4, label='rho_full', color='steelblue', alpha=0.85)
            axes3[0].bar(xx + 0.2, b, 0.4, label='rho_item', color='coral', alpha=0.85)
            for i in range(len(dfm)):
                axes3[0].plot([i - 0.2, i + 0.2], [a[i], b[i]], 'k-', linewidth=0.5, alpha=0.5)
            axes3[0].axhline(0, color='k', linewidth=0.8)
            axes3[0].set_xticks(xx)
            axes3[0].set_xticklabels(dfm['participant'], rotation=45, ha='right')
            try:
                _, p_pair = wilcoxon(a, b, alternative='two-sided')
            except Exception:
                p_pair = np.nan
            axes3[0].set_title(f'Wilcoxon rho_full vs rho_item p={p_pair:.4g}')
            axes3[1].scatter(a, b, s=45, edgecolors='k')
            lim = max(np.nanmax(np.abs(a)), np.nanmax(np.abs(b)), 1e-6)
            axes3[1].plot([-lim, lim], [-lim, lim], 'k--', alpha=0.6)
            axes3[1].set_xlabel('masked rho_full_diff')
            axes3[1].set_ylabel('masked rho_item_diff')
            plt.tight_layout()
            fig3.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig3_acc_hpc_variance_partition.pdf'), dpi=300, bbox_inches='tight')
            fig3.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig3_acc_hpc_variance_partition.png'), dpi=300, bbox_inches='tight')
            plt.show()

            cat_share = float(np.nanmean(a - b))
            item_part = float(np.nanmean(b))
            fig3b, axb = plt.subplots(figsize=(4, 4))
            axb.bar([0], [item_part], label='rho_item', color='coral')
            axb.bar([0], [cat_share], bottom=item_part, label='full−item', color='steelblue')
            axb.axhline(float(np.nanmean(a)), color='k', linestyle='--')
            axb.set_xticks([0])
            axb.set_xticklabels(['group'])
            axb.legend(fontsize=8)
            plt.tight_layout()
            fig3b.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig3b_acc_hpc_stacked_decomp.pdf'), dpi=300, bbox_inches='tight')
            fig3b.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig3b_acc_hpc_stacked_decomp.png'), dpi=300, bbox_inches='tight')
            plt.show()

            metrics = [
                ('masked_rho_full_diff', 'rho_full_diff'),
                ('masked_rho_item_diff', 'rho_item_diff'),
                ('masked_ri_cat_diff', 'ri_cat_diff'),
                ('masked_ri_item_diff', 'ri_item_diff'),
            ]
            fig4, axes4 = plt.subplots(2, 2, figsize=(12, 8))
            for ax, (col, name) in zip(axes4.ravel(), metrics):
                v = dfm[col].dropna().to_numpy(dtype=float)
                if len(v) == 0:
                    continue
                order = np.argsort(-v)
                cols = ['steelblue' if val >= 0 else 'salmon' for val in v[order]]
                ax.bar(np.arange(len(v)), v[order], color=cols, edgecolor='k', linewidth=0.4)
                ax.axhline(0, color='k', linewidth=0.8)
                ax.axhline(np.nanmean(v), color='coral', linestyle='--')
                ax.set_xticks(np.arange(len(v)))
                ax.set_xticklabels(np.asarray(dfm['participant'])[order], rotation=45, ha='right')
                ax.set_title(name)
                try:
                    _, p0 = wilcoxon(v, alternative='two-sided')
                    ax.text(0.02, 0.98, f'p={p0:.4g}', transform=ax.transAxes, va='top', fontsize=8)
                except Exception:
                    pass
            plt.tight_layout()
            fig4.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig4_acc_hpc_masked_subject_bars.pdf'), dpi=300, bbox_inches='tight')
            fig4.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig4_acc_hpc_masked_subject_bars.png'), dpi=300, bbox_inches='tight')
            plt.show()

            labels8 = ['rho_full', 'rho_item', 'ri_cat', 'ri_item']
            cols8 = [m[0] for m in metrics]
            means, lows, highs = [], [], []
            rng = np.random.default_rng(42)
            for col in cols8:
                v = dfm[col].dropna().to_numpy(dtype=float)
                if len(v) == 0:
                    means.append(np.nan)
                    lows.append(np.nan)
                    highs.append(np.nan)
                    continue
                means.append(float(np.mean(v)))
                boots = [rng.choice(v, size=len(v), replace=True).mean() for _ in range(1000)]
                boots = np.asarray(boots)
                lows.append(float(np.percentile(boots, 2.5)))
                highs.append(float(np.percentile(boots, 97.5)))
            fig8, ax8 = plt.subplots(figsize=(8, 4))
            x8 = np.arange(len(labels8))
            marr, larr, harr = np.array(means), np.array(lows), np.array(highs)
            err = [marr - larr, harr - marr]
            ax8.bar(x8, marr, yerr=err, capsize=4, color='steelblue', edgecolor='k')
            ax8.set_xticks(x8)
            ax8.set_xticklabels(labels8)
            ax8.axhline(0, color='k', linewidth=0.8)
            plt.tight_layout()
            fig8.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig8_acc_hpc_headline_bars.pdf'), dpi=300, bbox_inches='tight')
            fig8.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig8_acc_hpc_headline_bars.png'), dpi=300, bbox_inches='tight')
            plt.show()

        # ----- Figure 5: marginals
        rf = np.nanmean(stacks['rho_full_diff'], axis=0)
        inside = sig_mask_z & np.isfinite(rf)
        if np.any(inside):
            peak_idx = np.unravel_index(np.nanargmax(np.where(inside, rf, np.nan)), rf.shape)
        else:
            peak_idx = np.unravel_index(np.nanargmax(rf), rf.shape)
        t_fb_star, t_cue_star = int(peak_idx[0]), int(peak_idx[1])

        fig5, axes5 = plt.subplots(2, 1, figsize=(8, 8))
        for name in diff_names:
            D = stacks[name]
            if D is None:
                continue
            m = np.nanmean(D, axis=0)
            axes5[0].plot(fb_t, m[:, t_cue_star], label=name, alpha=0.85)
        axes5[0].axvline(fb_t[t_fb_star], color='k', linestyle=':', alpha=0.5)
        axes5[0].set_xlabel(f'Feedback ({ENC_ROI})')
        axes5[0].legend(fontsize=7)
        for name in diff_names:
            D = stacks[name]
            if D is None:
                continue
            m = np.nanmean(D, axis=0)
            axes5[1].plot(cue_t, m[t_fb_star, :], label=name, alpha=0.85)
        axes5[1].axvline(cue_t[t_cue_star], color='k', linestyle=':', alpha=0.5)
        axes5[1].set_xlabel(f'Cue ({CUE_ROI})')
        axes5[1].legend(fontsize=7)
        plt.tight_layout()
        fig5.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig5_acc_hpc_marginals.pdf'), dpi=300, bbox_inches='tight')
        fig5.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig5_acc_hpc_marginals.png'), dpi=300, bbox_inches='tight')
        plt.show()

        # ----- Figure 6: example subject RDMs
        try:
            ex_rank = rank_example_subjects_for_pair(
                trial_save_dir=RSA_TRIAL_SAVE_DIR,
                group_save_dir=RSA_GROUP_SAVE_DIR,
                roi_pair=(ENC_ROI, CUE_ROI),
            )
            if len(ex_rank) > 0:
                ex_subj = str(ex_rank.iloc[0]['subject'])
                s_obj = Subject(ex_subj, base_dir, load_dir, min_trials=MIN_TRIALS, verbose=False)
                ex_data = load_subject_crossregion_hfa(
                    subject_obj=s_obj,
                    encoding_roi=ENC_ROI,
                    cue_roi=CUE_ROI,
                    event_encoding=EVENT_ENCODING,
                    event_cue=EVENT_CUE,
                    freq_range=FREQ_RANGE,
                    downsample_factor=DOWNSAMPLE_FACTOR,
                    win_ms=WIN_MS,
                    min_trials=MIN_TRIALS,
                )
                rpe_ex = np.asarray(ex_data['rpe_categories'], dtype=object)
                hit_m = ex_data['memory_outcomes'] == 'remembered'
                labs = np.array([_canonicalize_rpe_cat_label(x) or 'na' for x in rpe_ex[hit_m]])
                order = np.argsort(pd.Categorical(labs, categories=['neg', 'neutral', 'pos', 'na']))
                enc_h = ex_data['encoding_hfa'][hit_m][order][:, :, t_fb_star]
                cue_h = ex_data['cue_hfa'][hit_m][order][:, :, t_cue_star]
                v_acc = _compute_pairwise_dissimilarity(enc_h, method=RDM_METHOD)
                v_hpc = _compute_pairwise_dissimilarity(cue_h, method=RDM_METHOD)
                rdm_acc = squareform(v_acc) if v_acc.size else np.zeros((0, 0))
                rdm_hpc = squareform(v_hpc) if v_hpc.size else np.zeros((0, 0))
                S_ex = _padded_row_zscore_cosine_similarity(enc_h, cue_h)
                M_ex = build_rpe_category_model_rdm(rpe_ex[hit_m][order])
                fig6, ax6 = plt.subplots(1, 4, figsize=(16, 3.5))
                vmax6 = max(np.nanmax(rdm_acc) if rdm_acc.size else 0, 1e-6)
                ax6[0].imshow(rdm_acc, cmap='viridis', vmin=0, vmax=vmax6)
                ax6[0].set_title('ACC RDM')
                ax6[1].imshow(rdm_hpc, cmap='viridis', vmin=0, vmax=vmax6)
                ax6[1].set_title('HPC RDM')
                ax6[2].imshow(S_ex, cmap='RdBu_r', vmin=-1, vmax=1)
                ax6[2].set_title('S')
                vmn = np.nanmin(M_ex)
                vmx = np.nanmax(M_ex)
                ax6[3].imshow(M_ex, cmap='Greys', vmin=vmn, vmax=vmx)
                ax6[3].set_title('M_rpe')
                for ax in ax6:
                    ax.set_xticks([])
                    ax.set_yticks([])
                plt.suptitle(f'{ex_subj} | sorted by rpe_cat', fontsize=10)
                plt.tight_layout()
                fig6.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig6_acc_hpc_example_rdms_category_sorted.pdf'), dpi=300, bbox_inches='tight')
                fig6.savefig(os.path.join(ACC_HPC_PLOT_DIR, 'fig6_acc_hpc_example_rdms_category_sorted.png'), dpi=300, bbox_inches='tight')
                plt.show()
        except Exception as e6:
            print('Figure 6 skipped:', e6)

        # ----- Figure 7: small multiples
        n_s = stacks['rho_item_diff'].shape[0]
        ncol = max(1, int(np.ceil(np.sqrt(n_s))))
        nrow = int(np.ceil(n_s / ncol))

        def _plot_small_mult(stack, name, fname_suffix):
            if n_s < 1:
                return
            vmax = _vmax99_list([np.nanmean(stack, axis=0)])
            fig, axes = plt.subplots(nrow, ncol, figsize=(2 * ncol, 2 * nrow))
            axes = np.atleast_2d(axes)
            for i in range(n_s):
                r, c = i // ncol, i % ncol
                ax = axes[r, c]
                g = stack[i]
                ax.imshow(
                    g, origin='lower', aspect='auto', extent=extent,
                    cmap='RdBu_r', vmin=-vmax, vmax=vmax,
                )
                if sig_mask_z.shape == g.shape:
                    ax.contour(cue_t, fb_t, sig_mask_z.astype(int), levels=[0.5], colors='k', linewidths=0.6)
                ax.set_title(loaded_p[i], fontsize=7)
                ax.set_xticks([])
                ax.set_yticks([])
            for j in range(n_s, nrow * ncol):
                rj, cj = j // ncol, j % ncol
                axes[rj, cj].axis('off')
            plt.suptitle(name, fontsize=11)
            plt.tight_layout()
            fig.savefig(os.path.join(ACC_HPC_PLOT_DIR, fname_suffix), dpi=300, bbox_inches='tight')
            fig.savefig(os.path.join(ACC_HPC_PLOT_DIR, fname_suffix.replace('.pdf', '.png')), dpi=300, bbox_inches='tight')
            plt.show()

        _plot_small_mult(stacks['rho_item_diff'], 'rho_item_diff', 'fig7_acc_hpc_smallmult_rho_item_diff.pdf')
        _plot_small_mult(stacks['ri_cat_diff'], 'ri_cat_diff', 'fig7_acc_hpc_smallmult_ri_cat_diff.pdf')

        # ----- Combined PDF from PNGs
        from matplotlib.backends.backend_pdf import PdfPages as _Pdf
        from matplotlib.image import imread
        out_comb = os.path.join(ACC_HPC_PLOT_DIR, 'acc_hpc_reactivation_content_report.pdf')
        pngs = sorted(glob.glob(os.path.join(ACC_HPC_PLOT_DIR, 'fig*.png')))
        if pngs:
            with _Pdf(out_comb) as pdf:
                for png in pngs:
                    im = imread(png)
                    fc, axc = plt.subplots(figsize=(12, 10))
                    axc.imshow(im)
                    axc.axis('off')
                    pdf.savefig(fc, bbox_inches='tight')
                    plt.close(fc)
            print('Saved', out_comb)
        else:
            # at least fig1 has png
            print('No fig*.png for combined PDF; save fig1 as png first')
