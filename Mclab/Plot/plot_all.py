import matplotlib.pyplot as plt
from plot_mclab import plot_IA_model

models = [
    '../Decision_Tree/mclab_tree.sav',
    #'../Lasso_Regression/mclab_lasso.sav',
    #'../Neural_Network/mclab_ann_best.sav',
    #'../Neural_Network/mclab_ann_small.sav',
    #'../Random_Forest/mclab_forest.sav',
    #'../SVR/mclab_svr_sigmoid.sav',
    #'../SVR/mclab_svr_rbf.sav',
    #'../SVR/mclab_svr_poly.sav',
    #'../SVR/mclab_svr_linear.sav',
    #'../XGBOOST/mclab_xgboost.sav',
]

titulos = ["Árvore de Decisão", "Regressão Lasso", 'Rede Neural (melhor)', 'Rede Neural (menor)',
           'Floresta Aleatória', 'SVR (Kernel Sigmoide)', 'SVR (Kernel RBF)', 'SVR (Kernel Polinomial)',
           'SVR (Kernel Linear)', 'XGBOOST']

for index, model in enumerate(models):
    plot_IA_model(model, titulos[index])
plt.show()