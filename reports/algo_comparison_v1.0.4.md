# Algorithm Comparison

Sorted by priority macro F1 desc then department macro F1.

| Algo | Params | Priority Macro F1 | Dept Macro F1 | Priority Acc | Dept Acc | Train (s) | Priority Extra |
|------|--------|-------------------|---------------|-------------|----------|-----------|---------------|
| logreg | {"C": 5.0, "class_weight": "balanced"} | 0.2654 | 0.3284 | 0.2653 | 0.4125 | 3.95 | False |
| linearsvc | {"C": 0.5} | 0.2611 | 0.2818 | 0.2611 | 0.5910 | 1.70 | False |
| linearsvc | {"C": 1.0} | 0.2597 | 0.2969 | 0.2597 | 0.5521 | 2.36 | True |
| linearsvc | {"C": 1.0} | 0.2591 | 0.2969 | 0.2590 | 0.5521 | 2.16 | False |
| logreg | {"C": 2.0, "class_weight": "balanced"} | 0.2588 | 0.3335 | 0.2590 | 0.4007 | 3.11 | False |
| linearsvc | {"C": 2.0} | 0.2587 | 0.3249 | 0.2583 | 0.5271 | 2.32 | False |
| linearsvc | {"C": 2.0} | 0.2549 | 0.3249 | 0.2549 | 0.5271 | 2.91 | True |
| linearsvc | {"C": 0.5} | 0.2536 | 0.2818 | 0.2535 | 0.5910 | 2.08 | True |
| logreg | {"C": 5.0, "class_weight": "balanced"} | 0.2534 | 0.3284 | 0.2535 | 0.4125 | 4.06 | True |
| logreg | {"C": 2.0, "class_weight": "balanced"} | 0.2438 | 0.3335 | 0.2437 | 0.4007 | 3.40 | True |
| logreg | {"C": 1.0, "class_weight": "balanced"} | 0.2436 | 0.3267 | 0.2437 | 0.3840 | 2.43 | False |
| mnb | {"alpha": 2.0} | 0.2422 | 0.2545 | 0.2528 | 0.6174 | 1.82 | True |
| logreg | {"C": 1.0, "class_weight": "balanced"} | 0.2402 | 0.3267 | 0.2403 | 0.3840 | 2.74 | True |
| mnb | {"alpha": 0.5} | 0.2401 | 0.2567 | 0.2437 | 0.6174 | 1.71 | True |
| mnb | {"alpha": 2.0} | 0.2394 | 0.2545 | 0.2486 | 0.6174 | 1.49 | False |
| logreg | {"C": 0.5, "class_weight": "balanced"} | 0.2388 | 0.3283 | 0.2396 | 0.3729 | 2.73 | True |
| mnb | {"alpha": 1.0} | 0.2388 | 0.2545 | 0.2431 | 0.6174 | 1.44 | False |
| logreg | {"C": 0.5, "class_weight": "balanced"} | 0.2379 | 0.3283 | 0.2382 | 0.3729 | 2.30 | False |
| mnb | {"alpha": 1.0} | 0.2356 | 0.2545 | 0.2410 | 0.6174 | 1.74 | True |
| mnb | {"alpha": 0.5} | 0.2347 | 0.2567 | 0.2368 | 0.6174 | 1.61 | False |

## Best Configurations
- Priority: logreg {'C': 5.0, 'class_weight': 'balanced'} priority_extra=False macro_f1=0.2654
- Department: logreg {'C': 2.0, 'class_weight': 'balanced'} priority_extra=False macro_f1=0.3335