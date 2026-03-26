# Experiment Matrix

- Rows: 12
- Updated (UTC): 2026-03-26T13:58:04Z

| Method | Attack | Seed | Epochs | SSL | PDB | Ratio | BA(before) | ASR(before) | BA(after) | ASR(after) | BA drop | ASR drop | Run |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| R | BadNets | 666 | 150 |  |  |  | 0.9141 | 0.9729 | 0.9010 | 0.1049 | 0.0131 | 0.8680 | experiments/badnets_refine/runs |
| R | BadNets | 666 | 50 | 0.02 | 0.5 | 0.5 | 0.9182 | 0.9731 | 0.8537 | 0.1147 | 0.0645 | 0.8584 | experiments/badnets_R_refine_e50_s666 |
| RB | BadNets | 666 | 150 | 0.02 | 0.15 | 0.3 | 0.9152 | 0.9731 | 0.8783 | 0.1033 | 0.0369 | 0.8698 | experiments/pdb_a_badnets_s666_refine_only_from_ckpt |
| RS | BadNets | 666 | 15 | 0.0 |  |  | 0.7960 | 0.9627 | 0.7739 | 0.1322 | 0.0221 | 0.8305 | experiments/quick_diag/ssl_w0/runs |
| RS | BadNets | 666 | 15 | 0.008 |  |  | 0.7960 | 0.9627 | 0.7398 | 0.1003 | 0.0562 | 0.8624 | experiments/quick_diag/ssl_w0008/runs |
| RS | BadNets | 666 | 30 | 0.004 |  |  | 0.7960 | 0.9627 | 0.8156 | 0.1095 | -0.0196 | 0.8532 | experiments/quick_diag/ssl_w0004_e30_v2/runs |
| RS | BadNets | 666 | 30 | 0.005 |  |  | 0.7960 | 0.9627 | 0.8204 | 0.1181 | -0.0244 | 0.8446 | experiments/quick_diag/ssl_w0005_e30_v2/runs |
| RS | BadNets | 777 | 30 | 0.004 |  |  | 0.8048 | 0.9615 | 0.8203 | 0.1165 | -0.0155 | 0.8450 | experiments/final/seed777/runs |
| RSB | BadNets | 666 | 150 | 0.001 | 0.05 | 0.1 | 0.9182 | 0.9731 | 0.8403 | 0.1039 | 0.0779 | 0.8692 | experiments/exp_refine_pdb_ssl_a_s666 |
| RSB | BadNets | 666 | 50 | 0.02 | 0.3 | 0.2 | 0.9182 | 0.9731 | 0.8414 | 0.1030 | 0.0768 | 0.8701 | experiments/badnets_Decoupled_refine_pdb_ssl_e50_s666 |
| RSB | BadNets | 666 | 50 | 0.02 | 0.3 | 0.2 | 0.9182 | 0.9731 | 0.8511 | 0.1153 | 0.0671 | 0.8578 | experiments/badnets_RSB_refine_pdb_ssl_e50_s666 |
| U | Unknown | 666 | 120 |  |  |  | 0.9054 | 0.9702 | 0.8976 | 0.1032 | 0.0078 | 0.8670 | experiments/runs_full_no_lc |
