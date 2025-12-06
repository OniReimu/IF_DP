# Ablation Results

## Base run (ε=2.0, k=2048, epochs=300, clip=2.0, rho_sat=0.001)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  36.0%     0.479
   Vanilla DP-SGD + DP-SAT         37.4%     0.498
   Fisher DP + Normal              46.0%     0.506
   Fisher DP + DP-SAT              49.9%     0.501
   Fisher DP + Normal + Calib      47.7%     0.495
   Fisher DP + DP-SAT + Calib      51.3%     0.505

## rho_sat = 0.005 (ε=2.0, k=2048, clip=2.0, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  36.0%     0.479
   Vanilla DP-SGD + DP-SAT         37.5%     0.498
   Fisher DP + Normal              46.0%     0.506
   Fisher DP + DP-SAT              50.1%     0.501
   Fisher DP + Normal + Calib      47.7%     0.495
   Fisher DP + DP-SAT + Calib      51.4%     0.505

## ε = 4.0 (k=512, clip=2.0, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  44.0%     0.486
   Vanilla DP-SGD + DP-SAT         44.3%     0.498
   Fisher DP + Normal              54.7%     0.508
   Fisher DP + DP-SAT              57.9%     0.499
   Fisher DP + Normal + Calib      56.1%     0.498
   Fisher DP + DP-SAT + Calib      59.0%     0.500

## ε = 8.0 (k=512, clip=2.0, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  48.5%     0.483
   Vanilla DP-SGD + DP-SAT         48.7%     0.495
   Fisher DP + Normal              56.9%     0.505
   Fisher DP + DP-SAT              59.4%     0.506
   Fisher DP + Normal + Calib      57.6%     0.503
   Fisher DP + DP-SAT + Calib      60.3%     0.497
## clip_radius = 1.0 (ε=2.0, k=2048, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  58.2%     0.488
   Vanilla DP-SGD + DP-SAT         58.2%     0.505
   Fisher DP + Normal              59.6%     0.510
   Fisher DP + DP-SAT              60.5%     0.507
   Fisher DP + Normal + Calib      60.2%     0.500
   Fisher DP + DP-SAT + Calib      61.1%     0.490

## clip_radius = 1.5 (ε=2.0, k=2048, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  49.2%     0.480
   Vanilla DP-SGD + DP-SAT         49.0%     0.495
   Fisher DP + Normal              54.8%     0.498
   Fisher DP + DP-SAT              57.5%     0.477
   Fisher DP + Normal + Calib      55.3%     0.506
   Fisher DP + DP-SAT + Calib      57.9%     0.495

## clip_radius = 2.5 (ε=2.0, k=2048, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  26.2%     0.482
   Vanilla DP-SGD + DP-SAT         30.0%     0.492
   Fisher DP + Normal              36.2%     0.496
   Fisher DP + DP-SAT              41.7%     0.487
   Fisher DP + Normal + Calib      36.9%     0.497
   Fisher DP + DP-SAT + Calib      42.5%     0.490

## clip_radius = 3.0 (ε=2.0, k=2048, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  21.3%     0.491
   Vanilla DP-SGD + DP-SAT         25.3%     0.489
   Fisher DP + Normal              29.7%     0.500
   Fisher DP + DP-SAT              35.1%     0.491
   Fisher DP + Normal + Calib      29.7%     0.498
   Fisher DP + DP-SAT + Calib      35.4%     0.485

## k = 256 (ε=2.0, clip=2.0, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  36.0%     0.479
   Vanilla DP-SGD + DP-SAT         37.4%     0.498
   Fisher DP + Normal              50.2%     0.498
   Fisher DP + DP-SAT              54.3%     0.501
   Fisher DP + Normal + Calib      52.3%     0.504
   Fisher DP + DP-SAT + Calib      56.2%     0.512

## k = 512 (ε=2.0, clip=2.0, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  36.0%     0.479
   Vanilla DP-SGD + DP-SAT         37.4%     0.498
   Fisher DP + Normal              50.0%     0.514
   Fisher DP + DP-SAT              54.7%     0.500
   Fisher DP + Normal + Calib      51.9%     0.497
   Fisher DP + DP-SAT + Calib      55.9%     0.501

## k = 1024 (ε=2.0, clip=2.0, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  36.0%     0.479
   Vanilla DP-SGD + DP-SAT         37.4%     0.498
   Fisher DP + Normal              49.8%     0.492
   Fisher DP + DP-SAT              52.9%     0.515
   Fisher DP + Normal + Calib      51.3%     0.499
   Fisher DP + DP-SAT + Calib      54.3%     0.506

## epochs = 150 (ε=2.0, k=2048, clip=2.0)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  41.1%     0.498
   Vanilla DP-SGD + DP-SAT         37.4%     0.501
   Fisher DP + Normal              50.6%     0.493
   Fisher DP + DP-SAT              51.3%     0.505
   Fisher DP + Normal + Calib      54.0%     0.504
   Fisher DP + DP-SAT + Calib      54.1%     0.495

## epochs = 200 (ε=2.0, k=2048, clip=2.0)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  39.0%     0.487
   Vanilla DP-SGD + DP-SAT         37.0%     0.486
   Fisher DP + Normal              48.9%     0.503
   Fisher DP + DP-SAT              49.7%     0.523
   Fisher DP + Normal + Calib      50.3%     0.504
   Fisher DP + DP-SAT + Calib      50.5%     0.507

## rho_sat = 0.0005 (ε=2.0, k=2048, clip=2.0, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  36.0%     0.479
   Vanilla DP-SGD + DP-SAT         37.5%     0.498
   Fisher DP + Normal              46.0%     0.506
   Fisher DP + DP-SAT              49.9%     0.501
   Fisher DP + Normal + Calib      47.7%     0.495
   Fisher DP + DP-SAT + Calib      51.3%     0.505

## rho_sat = 0.002 (ε=2.0, k=2048, clip=2.0, epochs=300)
⚖️  Privacy vs Accuracy Tradeoff:
   Model                          Accuracy  Privacy (1-AUC)
   ────────────────────────────── ─────────  ──────────────
   Vanilla DP-SGD                  36.0%     0.479
   Vanilla DP-SGD + DP-SAT         37.4%     0.498
   Fisher DP + Normal              46.0%     0.506
   Fisher DP + DP-SAT              49.9%     0.501
   Fisher DP + Normal + Calib      47.7%     0.495
   Fisher DP + DP-SAT + Calib      51.3%     0.505
