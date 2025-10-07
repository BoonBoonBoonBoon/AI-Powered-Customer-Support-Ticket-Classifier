# Error Analysis: v1.0.3 (validation split)

## Priority
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| High | 0.605 | 0.609 | 0.607 | 417 |
| Low | 0.611 | 0.644 | 0.627 | 413 |
| Medium | 0.676 | 0.632 | 0.653 | 438 |
| Urgent | 0.655 | 0.660 | 0.657 | 426 |
| Macro Avg | 0.637 | 0.636 | 0.636 | - |

## Department
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Billing | 0.883 | 0.671 | 0.762 | 1020 |
| Sales | 0.565 | 0.772 | 0.652 | 333 |
| Tech Support | 0.547 | 0.745 | 0.631 | 341 |
| Macro Avg | 0.665 | 0.729 | 0.682 | - |

## High-Confidence Misclassifications (confidence >= 0.6)
| Title | P_true | P_pred | P_conf | D_true | D_pred | D_conf | MisP | MisD |
|-------|--------|--------|--------|--------|--------|--------|------|------|
| Hardware issue | Medium | Medium | 0.35 | Billing | Sales | 0.62 |  | Y |
| Software bug | Low | Low | 0.32 | Billing | Sales | 0.61 |  | Y |