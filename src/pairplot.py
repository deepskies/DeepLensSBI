from sbi import utils

fig, ax = utils.pairplot(samples, 
                             points=[true_parameter,best_fit_t],
                             labels=[r'$\theta_E$',r'$le1$',r'$1e2$', r'smag',r'x',r'y',r'R',r'n',r'se1',r'se2'], 
                             limits=limits,
                             points_colors=['r','b'],
                             points_offdiag={'markersize': 6},
                             fig_size=[12, 12])
