                Machine Learning Algorithms
                        |
           -------------------------------------
           |                                   |
     Supervised                          Unsupervised
           |                                   |
    ----------------                  -----------------
    |              |                  |               |
Classification    Regression        Clustering      Dimensionality Reduction
    |              |                  |               |
  --------       --------           ------------      -------------------
  |      |       |      |           |          |      |                 |
Linear  Tree-based   Support      K-Means   Hierarchical   Principal Component
Models    Models   Vector Machine   Clustering  Clustering       Analysis
  |        |          (SVM)                     |             (PCA)
  |        |                                   |
  ----     -----                           -----
  |  |     |   |                           |   |
Lasso  Random Gradient                 K-Nearest   Gaussian Mixture
Regression  Boosting                     Neighbors    Models
            |                            (KNN)         |
        --------                                     ------
        |      |                                     |    |
Decision Trees  Gradient                         DBSCAN   Expectation
                Boosting                                 Maximization
                                                         (EM) Algorithm

----------------------------------------------------------------------------------------------------------------------


                  Kernelized Methods
                         |
            --------------------------------------
            |                                     |
   Kernel Support Vector Machines       Kernel Methods for Dimensionality Reduction
            |                                     |
    -----------------------------       ----------------------------
    |                           |       |                          |
Gaussian Radial Basis Function   Polynomial          Kernel Principal Component Analysis
           Kernel                Kernel                             (Kernel PCA)
    (RBF Kernel)                  |
                                 ------
                                 |    |
                            Laplacian   Polynomial
                            Kernel      Kernel
                             (LK)        (PK)

----------------------------------------------------------------------------------------------------------------------
                 Ensemble Methods
                        |
          ---------------------------------
          |                               |
     Bagging                       Boosting
          |                               |
   -------------------         -------------------
   |                 |         |                 |
Random Forests    Extra Trees  AdaBoost      Gradient Boosting
                    |                        |
         ---------------             -------------------
         |             |             |                 |
   XGBoost       LightGBM     CatBoost        Stacking
                                      |
                              ----------------
                              |              |
                       Voting Classifier  Stacking Regressor
----------------------------------------------------------------------------------------------------------------------
               Evaluation Methods
                       |
       -------------------------------
       |                             |
Supervised Learning          Unsupervised Learning
       |                             |
  ------------------         ------------------
  |                |         |                |
Classification   Regression   Clustering   Dimensionality Reduction
       |                |         |                |
    ---------      ---------  -----------------  ------------------
    |       |      |       |  |                 |  |                |
Accuracy  F1-Score  RMSE   MAE  Silhouette Score  Reconstruction Error
   |       |         |       |                   |
Precision  Recall   R-Squared               Explained Variance Score
   |       |         |       |                   |
ROC Curve  PR Curve  |       |                   |
           |        ---------              ----------
           |        |       |              |        |
         Confusion  Precision-           Elbow   Scree Plot
           Matrix   Recall Curve          Method

----------------------------------------------------------------------------------------------------------------------

                Data Partitions
                       |
       -------------------------------
       |                             |
       |                             |
   Training Data              Testing Data
       |                             |
------------------         ------------------
|                |         |                |
Train-Validation   Holdout    Cross-Validation
   Split           Split
                     |
           -------------------
           |        |        |
   K-Fold CV  Stratified CV  Leave-One-Out
                     |
              -----------------
              |               |
        Shuffle-Split   Time Series Split
----------------------------------------------------------------------------------------------------------------------