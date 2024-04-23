import matplotlib.pyplot as plt
from plot_mclab import plot_IA_model, plot_classic_models
import classic_models

modelos_ia = [
    '../Neural_Network/mclab_ann_best.sav',
    '../Neural_Network/mclab_ann_small.sav',
    '../Decision_Tree/mclab_tree.sav',
    '../Lasso_Regression/mclab_lasso.sav',
    '../Random_Forest/mclab_forest.sav',
    '../SVR/mclab_svr_sigmoid.sav',
    '../SVR/mclab_svr_rbf.sav',
    '../SVR/mclab_svr_poly.sav',
    '../SVR/mclab_svr_linear.sav',
    '../XGBOOST/mclab_xgboost.sav',
]

titulos_IA = ['Rede Neural (melhor)', 'Rede Neural (menor)', "Árvore de Decisão", "Regressão Lasso",
           'Floresta Aleatória', 'SVR (Kernel Sigmoide)', 'SVR (Kernel RBF)', 'SVR (Kernel Polinomial)',
           'SVR (Kernel Linear)', 'XGBOOST']

for index, model in enumerate(modelos_ia):
    print(titulos_IA[index])
    plot_IA_model(model, titulos_IA[index])


modelos_classicos = [classic_models.path_loss_free_space, classic_models.path_loss_egli,
                     classic_models.path_loss_okumura_hata, classic_models.path_loss_cost231_hata,
                     classic_models.path_loss_log_distance]

isLogDistance = [False, False, False, False, True]

titulos_classicos = ['Free Space', 'Egli', 'Okumura Hata', 'Cost231 Hata', 'Log Distance']


for index, model in enumerate(modelos_classicos):
    print(titulos_classicos[index])
    plot_classic_models(model, titulos_classicos[index], isLogDistance[index])



plt.show()