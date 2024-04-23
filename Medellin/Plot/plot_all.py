import matplotlib.pyplot as plt
from plot_medellin import plot_IA_model

models_all = [
    '../Neural_Network/medellin_ann_relevant.sav',
    '../Decision_Tree/medellin_tree_relevant.sav',
    '../Lasso_Regression/medellin_lasso_relevant.sav',
    '../Random_Forest/medellin_forest_relevant.sav',
    '../SVR/medellin_svr_sigmoid_relevant.sav',
    '../SVR/medellin_svr_rbf_relevant.sav',
    '../SVR/medellin_svr_poly_relevant.sav',
    '../SVR/medellin_svr_linear_relevant.sav',
    '../XGBOOST/medellin_xgboost_relevant.sav',
]

titulos = ['Rede Neural', "Árvore de Decisão", "Regressão Lasso",
           'Floresta Aleatória', 'SVR (Kernel Sigmoide)', 'SVR (Kernel RBF)', 'SVR (Kernel Polinomial)',
           'SVR (Kernel Linear)', 'XGBOOST']

for index, model in enumerate(models_all):
    print(titulos[index])
    plot_IA_model(model, titulos[index])
plt.show()