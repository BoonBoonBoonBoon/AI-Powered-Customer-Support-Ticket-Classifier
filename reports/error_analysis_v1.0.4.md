# Error Analysis: v1.0.4 (validation split)

## Priority
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| High | 0.239 | 0.263 | 0.250 | 354 |
| Low | 0.252 | 0.262 | 0.257 | 351 |
| Medium | 0.260 | 0.231 | 0.244 | 373 |
| Urgent | 0.211 | 0.207 | 0.209 | 362 |
| Macro Avg | 0.241 | 0.241 | 0.240 | - |

## Department
| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|----|---------|
| Billing | 0.609 | 0.448 | 0.516 | 889 |
| Sales | 0.212 | 0.310 | 0.252 | 268 |
| Tech Support | 0.183 | 0.254 | 0.213 | 283 |
| Macro Avg | 0.334 | 0.337 | 0.327 | - |

## High-Confidence Misclassifications (confidence >= 0.6)
| Title | P_true | P_pred | P_conf | D_true | D_pred | D_conf | MisP | MisD |
|-------|--------|--------|--------|--------|--------|--------|------|------|
| Product recommendation | Medium | Low | 0.31 | Billing | Sales | 0.68 | Y | Y |
| Network problem | Medium | Low | 0.31 | Billing | Sales | 0.66 | Y | Y |
| Software bug | High | Urgent | 0.35 | Billing | Sales | 0.67 | Y | Y |
| Account access | High | Medium | 0.40 | Tech Support | Billing | 0.65 | Y | Y |
| Product setup | Urgent | Medium | 0.40 | Sales | Billing | 0.61 | Y | Y |
| Product compatibility | Urgent | Low | 0.32 | Billing | Sales | 0.64 | Y | Y |
| Delivery problem | High | Urgent | 0.26 | Billing | Sales | 0.62 | Y | Y |
| Software bug | Medium | Medium | 0.39 | Sales | Billing | 0.61 |  | Y |
| Data loss | High | Urgent | 0.30 | Billing | Tech Support | 0.60 | Y | Y |
| Cancellation request | Urgent | Low | 0.28 | Billing | Sales | 0.64 | Y | Y |