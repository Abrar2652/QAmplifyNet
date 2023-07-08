# QAmplifyNet
A Hybrid Quantum-Classical Neural Network for Supply Chain Backorder Prediction

-------
The code repository for the *Nature Scientific Reports* (2023) paper titled **"QAmplifyNet: Pushing the Boundaries of Supply Chain Backorder Prediction Using Interpretable Hybrid Quantum–Classical Neural Network"**.

![Visualization of proposed model](https://github.com/Abrar2652/QAmplifyNet/blob/main/QAmplifyNet/Proposed_framework.png)


Created and maintained by Md Abrar Jahin `<abrar.jahin.2652@gmail.com, md-jahin@oist.jp>`.

-------
## Benchmark Dataset

[*"Can you predict product backorder?"* dataset](https://www.kaggle.com/datasets/gowthammiryala/back-order-prediction-dataset)
-------
## Directory Tour

Below is an illustration of the directory structure of QAmplifyNet.

```
📁 QAmplifyNet
└── 📁 Classical
    📁 QAmplifyNet\Classical
    ├── 📁 ANN
    │   📁 QAmplifyNet\Classical\ANN
    │   ├── 📄 classification_report.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 model_architecture.png
    │   ├── 📄 model_summary.png
    │   ├── 📄 nn_roc_auc.png
    ├── 📁 Adacost
    │   📁 QAmplifyNet\Classical\Adacost
    │   ├── 📄 ada_confusion_matrix.png
    │   ├── 📄 ada_roc_auc.png
    │   ├── 📄 best_params.png
    │   ├── 📄 classification_report1.png
    │   ├── 📄 classification_report2.png
    ├── 📁 Catboost
    │   📁 QAmplifyNet\Classical\Catboost
    │   ├── 📄 best_params.png
    │   ├── 📄 cat_confusion_matrix.png
    │   ├── 📄 cat_roc_auc.png
    │   ├── 📄 catboost_feature_imp.png
    │   ├── 📄 classification_report.png
    ├── 📁 Decision Tree
    │   📁 QAmplifyNet\Classical\Decision Tree
    │   ├── 📄 classification_report_best_params.png
    │   ├── 📄 dt_confusion_matrix.png
    │   ├── 📄 dt_roc_auc.png
    ├── 📁 KNN
    │   📁 QAmplifyNet\Classical\KNN
    │   ├── 📄 best_params.png
    │   ├── 📄 classification_report.png
    │   ├── 📄 knn_confusion_matrix.png
    │   ├── 📄 knn_roc_auc.png
    ├── 📁 LGBM
    │   📁 QAmplifyNet\Classical\LGBM
    │   ├── 📄 best_params.png
    │   ├── 📄 classification_report.png
    │   ├── 📄 lgbm_confusion_matrix.png
    │   ├── 📄 lgbm_feature_imp.png
    │   ├── 📄 lgbm_roc_auc.png
    ├── 📁 Logistic Regression
    │   📁 QAmplifyNet\Classical\Logistic Regression
    │   ├── 📄 best_params.png
    │   ├── 📄 classification_report.png
    │   ├── 📄 log_confusion_matrix.png
    │   ├── 📄 log_roc_auc.png
    ├── 📄 PR_10_classical.png
    ├── 📁 RF
    │   📁 QAmplifyNet\Classical\RF
    │   ├── 📄 classification_report_best_params.png
    │   ├── 📄 rf_confusion_matrix.png
    │   ├── 📄 rf_roc_auc.png
    ├── 📄 ROC_10_classical.png
    ├── 📁 SVM
    │   📁 QAmplifyNet\Classical\SVM
    │   ├── 📄 best_params.png
    │   ├── 📄 classification_report.png
    │   ├── 📄 svm_confusion_matrix.png
    │   ├── 📄 svm_roc_auc.png
    ├── 📁 XGBoost
    │   📁 QAmplifyNet\Classical\XGBoost
    │   ├── 📄 classification_report_best_params.png
    │   ├── 📄 xgb_confusion_matrix.png
    │   ├── 📄 xgb_imp_features.png
    │   ├── 📄 xgb_roc_auc.png
    │   ├── 📄 xgboost_feature_imp.png
    ├── 📄 classical_modelling_short_data.ipynb
    ├── 📄 cm_10_classical.png
    ├── 📄 data_distribution.png
└── 📁 DDQN_RL
    📁 QAmplifyNet\DDQN_RL
    ├── 📄 DDQN_RL.ipynb
    ├── 📄 classification_reports.png
    ├── 📄 confusion_matrix.png
    ├── 📄 pr_curve.png
    ├── 📄 roc_curve.png
└── 📄 LICENSE
└── 📁 Preprocessing
    📁 QAmplifyNet\Preprocessing
    ├── 📁 IFLOF (except VIF)
    │   📁 QAmplifyNet\Preprocessing\IFLOF (except VIF)
    │   ├── 📄 nearmiss_lr.png
    ├── 📁 IFLOF+vif
    │   📁 QAmplifyNet\Preprocessing\IFLOF+vif
    │   ├── 📄 nearmiss_lr.png
    ├── 📁 IQR+vif
    │   📁 QAmplifyNet\Preprocessing\IQR+vif
    │   ├── 📄 nearmiss_lr1.png
    │   ├── 📄 nearmiss_lr2.png
    ├── 📁 Log transform+StandardScaler+VIF
    │   📁 QAmplifyNet\Preprocessing\Log transform+StandardScaler+VIF
    │   ├── 📄 nearmiss_lr.png
    ├── 📁 No Log transform+VIF
    │   📁 QAmplifyNet\Preprocessing\No Log transform+VIF
    │   ├── 📄 nearmiss_lr.png
    ├── 📄 Preprocessing.ipynb
    ├── 📁 RobScalar+VIF
    │   📁 QAmplifyNet\Preprocessing\RobScalar+VIF
    │   ├── 📄 nearmiss_lr.png
    ├── 📁 VIF+no outlier detector
    │   📁 QAmplifyNet\Preprocessing\VIF+no outlier detector
    │   ├── 📄 nearmiss_lr.png
    ├── 📄 after_anomaly_removal.png
    ├── 📄 after_null_data_removal.png
    ├── 📄 backorder_distributiom.png
    ├── 📄 backorder_distribution2.png
    ├── 📄 before_anomaly_removal.png
    ├── 📄 before_null_data_removal.png
    ├── 📄 correlation_heatmap.png
└── 📁 QAmplifyNet
    📁 QAmplifyNet\QAmplifyNet
    ├── 📁 3_dense_layers
    │   📁 QAmplifyNet\QAmplifyNet\3_dense_layers
    │   ├── 📄 PR_curve.png
    │   ├── 📄 QAmplifyNet.ipynb
    │   ├── 📄 ROC_curve.png
    │   ├── 📄 StronglyEntanglingLayers.png
    │   ├── 📄 accuracy.png
    │   ├── 📄 classification_reports1.png
    │   ├── 📄 classification_reports2.png
    │   ├── 📄 cm.png
    │   ├── 📄 data_shape.png
    │   ├── 📄 lime_explanation.png
    │   ├── 📄 lime_explanation_idx1.png
    │   ├── 📄 loss.png
    │   ├── 📄 loss_prc_recall.png
    │   ├── 📄 model_architecture1.png
    │   ├── 📄 model_architecture2.png
    │   ├── 📄 model_summary.png
    │   ├── 📄 shap_decision_plot.png
    │   ├── 📄 shap_test_idx20.png
    │   ├── 📄 shap_train_idx20.png
    │   ├── 📄 train_val_loss.png
    ├── 📄 Proposed_framework.png
    ├── 📄 qml.drawio
└── 📁 QEnsembles
    📁 QAmplifyNet\QEnsembles
    ├── 📁 LGBM_qSVM_as_base_LR_meta_stacking
    │   📁 QAmplifyNet\QEnsembles\LGBM_qSVM_as_base_LR_meta_stacking
    │   ├── 📄 classification_report.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 roc_curve.png
    ├── 📄 QEnsemble.ipynb
    ├── 📁 VQC_base_LGBM_meta_stacking
    │   📁 QAmplifyNet\QEnsembles\VQC_base_LGBM_meta_stacking
    │   ├── 📄 classification_reports.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 roc_curve.png
    ├── 📁 VQC_base_qSVM_metaclassifier_stacking
    │   📁 QAmplifyNet\QEnsembles\VQC_base_qSVM_metaclassifier_stacking
    │   └── 📄 classification_reports.png
    │   └── 📄 confusion_matrix.png
    │   └── 📄 roc_curve.png
└── 📁 QNN
    📁 QAmplifyNet\QNN
    ├── 📁 CNN+Encoder+QNN
    │   📁 QAmplifyNet\QNN\CNN+Encoder+QNN
    │   ├── 📄 NN_Encoder_QNN.ipynb
    │   ├── 📁 PCA_datasets
    │   │   📁 QAmplifyNet\QNN\CNN+Encoder+QNN\PCA_datasets
    │   │   ├── 📄 Xmiss_train_pca.csv
    │   │   ├── 📄 qtest_df_pca.csv
    │   │   ├── 📄 ymiss_train_pca.csv
    │   ├── 📄 QNN.pdf
    │   ├── 📄 classification_report_metrics.png
    │   ├── 📄 confusion_matrix.png
    │   ├── 📄 metrics_plots_generation.ipynb
    │   ├── 📁 outputs
    │   │   📁 QAmplifyNet\QNN\CNN+Encoder+QNN\outputs
    │   │   ├── 📁 confusion
    │   │   │   📁 QAmplifyNet\QNN\CNN+Encoder+QNN\outputs\confusion
    │   │   │   ├── 📁 1
    │   │   │   │   📁 QAmplifyNet\QNN\CNN+Encoder+QNN\outputs\confusion\1
    │   │   │   │   └── 📄 confusion_table.npy
    │   │   ├── 📁 models
    │   │   │   📁 QAmplifyNet\QNN\CNN+Encoder+QNN\outputs\models
    │   │   │   ├── 📁 1
    │   │   │   │   📁 QAmplifyNet\QNN\CNN+Encoder+QNN\outputs\models\1
    │   │   │   │   └── 📄 checkpoint
    │   │   │   │   └── 📄 optimum.txt
    │   │   │   │   └── 📄 sess.ckpt-26000.data-00000-of-00001
    │   │   │   │   └── 📄 sess.ckpt-26000.index
    │   │   │   │   └── 📄 sess.ckpt-26000.meta
    │   │   │   │   └── 📄 sess.ckpt-27000.data-00000-of-00001
    │   │   │   │   └── 📄 sess.ckpt-27000.index
    │   │   │   │   └── 📄 sess.ckpt-27000.meta
    │   │   │   │   └── 📄 sess.ckpt-28000.data-00000-of-00001
    │   │   │   │   └── 📄 sess.ckpt-28000.index
    │   │   │   │   └── 📄 sess.ckpt-28000.meta
    │   │   │   │   └── 📄 sess.ckpt-29000.data-00000-of-00001
    │   │   │   │   └── 📄 sess.ckpt-29000.index
    │   │   │   │   └── 📄 sess.ckpt-29000.meta
    │   │   │   │   └── 📄 sess.ckpt-30000.data-00000-of-00001
    │   │   │   │   └── 📄 sess.ckpt-30000.index
    │   │   │   │   └── 📄 sess.ckpt-30000.meta
    │   │   ├── 📁 tensorboard
    │   │   │   📁 QAmplifyNet\QNN\CNN+Encoder+QNN\outputs\tensorboard
    │   │   │   └── 📁 1
    │   │   │       📁 QAmplifyNet\QNN\CNN+Encoder+QNN\outputs\tensorboard\1
    │   │   │       └── 📄 events.out.tfevents.1686500392.DESKTOP-B9RPDDT
    │   │   │       └── 📄 events.out.tfevents.1686500812.DESKTOP-B9RPDDT
    │   ├── 📄 roc1.png
    │   ├── 📄 roc2.png
    ├── 📁 MERA
    │   📁 QAmplifyNet\QNN\MERA
    │   ├── 📁 MERA_1_layered
    │   │   📁 QAmplifyNet\QNN\MERA\MERA_1_layered
    │   │   ├── 📄 classification_report1.png
    │   │   ├── 📄 classification_report2.png
    │   │   ├── 📄 confusion_matrix.png
    │   │   ├── 📄 mera-1-layered.ipynb
    │   │   ├── 📄 model_architecture.png
    │   │   ├── 📄 roc_auc.png
    │   │   ├── 📄 train_val_accuracy.png
    │   ├── 📁 MERA_2_layered
    │   │   📁 QAmplifyNet\QNN\MERA\MERA_2_layered
    │   │   ├── 📄 classification_report1.png
    │   │   ├── 📄 classification_report2.png
    │   │   ├── 📄 confusion_matrix.png
    │   │   ├── 📄 mera-2-layered.ipynb
    │   │   ├── 📄 model_architecture.png
    │   │   ├── 📄 roc_auc.png
    │   │   ├── 📄 train_val_accuracy.png
    │   ├── 📁 MERA_4_layered
    │   │   📁 QAmplifyNet\QNN\MERA\MERA_4_layered
    │   │   └── 📄 classification_report1.png
    │   │   └── 📄 classification_report2.png
    │   │   └── 📄 confusion_matrix.png
    │   │   └── 📄 mera-4-layered.ipynb
    │   │   └── 📄 model_architecture.png
    │   │   └── 📄 roc_auc.png
    │   │   └── 📄 train_val_accuracy.png
    ├── 📁 RY-CNOT
    │   📁 QAmplifyNet\QNN\RY-CNOT
    │   └── 📄 architecture.png
    │   └── 📄 classificaion_report1.png
    │   └── 📄 classificaion_report2.png
    │   └── 📄 confusion_matrix.png
    │   └── 📄 roc_auc.png
    │   └── 📄 ry-cnot-vqc.ipynb
    │   └── 📄 train_val_accuracy.png
└── 📄 README.md
└── 📄 sc-ml-classical.ipynb

```

------------

## License

MIT licensed, except where otherwise stated.
See `LICENSE.txt` file.

















