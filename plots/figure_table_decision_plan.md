# Figure and table decision pack

## Main-report essentials

1. Clean 8 s performance table (`display_table_clean_8s.csv`).
   Use this as the main result table. It directly answers which backbone is strongest under the selected operating window.

2. Window-length trend figure.
   Use the existing `clean_f1_by_window_6_8_10.png` and/or `clean_auroc_by_window_6_8_10.png`.
   If only one fits, prefer F1 because it is the validation-threshold operating-point metric.

3. Accuracy--efficiency trade-off.
   Use the existing `clean_f1_vs_cpu_record_latency_focus_8s.png`, but discuss it with `display_table_efficiency_8s.csv`.
   This supports the practical at-home ECG device narrative.

4. Zero-shot robustness.
   Use the existing F1-vs-SNR plots for the three noise types. Add at least one specificity plot when discussing false positives under severe noise.
   Specificity is essential because several models keep high sensitivity under severe noise by predicting almost everything as AF.

5. Severe-noise summary table (`display_table_zero_shot_8s_severe_neg6.csv`).
   This is the compact table that makes the max-pooling/noise-vulnerability argument defensible.

## Strong appendix tables

- Full clean results across 4/6/8/10 s: `clean_summary.csv`
- Robustness summary by model/noise: `display_table_zero_shot_8s_robustness.csv`
- Full comparable efficiency table: `display_table_efficiency_8s.csv` plus `efficiency_context_length_summary_filtered.csv`
- Noisy-input training summary: `display_table_noisy_training_4s_robustness.csv`
- Zero-shot vs noisy-input training at 4 s, -6 dB: `display_table_zero_shot_vs_noisy_training_4s_neg6.csv`

## Plots added in this pack

- Specificity-vs-SNR at 8 s for bw/em/ma.
- AUROC-vs-SNR at 8 s for bw/em/ma.
- F1 drop from clean to -6 dB for bw/em/ma.
- 4 s zero-shot vs noisy-input training F1 at -6 dB for bw/em/ma.
