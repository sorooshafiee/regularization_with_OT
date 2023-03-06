import numpy as np
from scipy.sparse import vstack
from sklearn.datasets import load_svmlight_file
from sklearn.datasets import make_blobs
import gurobipy as grb
import matplotlib
import matplotlib.pyplot as plt
from matplotlib import rc
import matplotlib.patches as patches

rc('font', family='serif')
rc('text', usetex=True)


def gen_image(X, distributions, epsilons, norm):
    fig, ax_arr = plt.subplots(epsilons.size, figsize = (7.3,6.5))
    plt.rcParams['text.latex.preamble'] = r"\usepackage{bm} \usepackage{array} \usepackage{color}"
    n_samples = X.shape[0]
    
    for index, ax in enumerate(ax_arr):
        
        ax.set_title(r'$\varepsilon={0:g}$'.format(epsilons[index]),
                     fontdict={'size': 16})
        
        images = X.copy()
        
        if epsilons[index] != 0:
            xi, q = distributions[(norm, epsilons[index])]
            q_string = ''
            for i in range(n_samples):
                tmp = np.ceil(q * 100)
                if tmp[i] == 0:
                    q_string += r'{\textcolor{white}{100\%}} &'
                else:
                    q_string += str(tmp[i])[:-2] + r'\% &'
            alignment = r'\\' + r'{\textcolor{white}{*****}} &' * n_samples
            table2 = r'\begin{tabular}{cccccccccc}' + q_string[:-1] + alignment[:-1] + r' \end{tabular}'
            ax.text(0, 45, table2, fontsize = 12)
            for i in range(n_samples):
                images[i] = images[i] - xi[i] / (q[i] + 1e-10)
        
        three_d = np.reshape(1 - images, (-1, 28, 28)) * 255
        two_d = np.hstack([three_d[i, :, :].astype(np.uint8)
                           for i in range(three_d.shape[0])])
            
        ax.set_xticks([])
        ax.set_yticks([])
        ax.imshow(two_d, cmap='gray')
    return fig, ax_arr


def least_favorable_distribution(X, y, epsilon=0.0, norm=1, verbose=False):
    n_samples, n_features = X.shape
    model = grb.Model('LeastFavorable')
    model.setParam('OutputFlag', verbose)
    model.setAttr('ModelSense', grb.GRB.MAXIMIZE)

    var_q = model.addVars(
        n_samples,
        vtype=grb.GRB.CONTINUOUS,
        obj=1/n_samples,
        lb=0,
        ub=1,
        name='q'
    )
    var_xi = model.addVars(
        n_samples, n_features,
        vtype=grb.GRB.CONTINUOUS,
        lb=-grb.GRB.INFINITY,
        name='xi'
    )

    model.addConstrs(
        (grb.quicksum(
            var_xi[i, j] for i in range(n_samples)
        ) == grb.quicksum(
            var_q[i] * y[i] * X[i, j] for i in range(n_samples))
         for j in range(n_features))
    )
    model.addConstrs(
        (var_q[i] * X[i, j] - var_xi[i, j] >= 0)
        for i in range(n_samples)
        for j in range(n_features)
    )
    model.addConstrs(
        (var_q[i] * X[i, j] - var_xi[i, j] <= var_q[i])
        for i in range(n_samples)
        for j in range(n_features)
    )

    if norm == 1:
        slack_var = model.addVars(
            n_samples, n_features,
            vtype=grb.GRB.CONTINUOUS
        )
        model.addConstrs(
            (var_xi[i, j] <= slack_var[i, j]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstrs(
            (-var_xi[i, j] <= slack_var[i, j]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstr(
            grb.quicksum(slack_var[i, j]
                         for i in range(n_samples)
                         for j in range(n_features)) <= n_samples * epsilon
        )

    elif norm == 2:
        slack_var = model.addVars(
            n_samples,
            vtype=grb.GRB.CONTINUOUS
        )
        for i in range(n_samples):
            model.addQConstr(
                grb.quicksum(var_xi[i, j] * var_xi[i, j]
                             for j in range(n_features)) <= slack_var[i] * slack_var[i]
            )
        model.addConstr(
            grb.quicksum(slack_var[i]
                         for i in range(n_samples)) <= n_samples * epsilon
        )

    elif np.isinf(norm):
        slack_var = model.addVars(
            n_samples,
            vtype=grb.GRB.CONTINUOUS
        )
        model.addConstrs(
            (var_xi[i, j] <= slack_var[i]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstrs(
            (-var_xi[i, j] <= slack_var[i]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstr(
            grb.quicksum(slack_var[i]
                         for i in range(n_samples)) <= n_samples * epsilon
        )

    model.optimize()
    q = np.array([value.x for value in var_q.values()])
    xi = np.array([value.x for value in var_xi.values()])
    xi = np.reshape(xi, (n_samples, n_features))

    return xi, q


def worst_case_distribution(X, y, epsilon=0.0, qnorm=1, verbose=False):
    n_samples, n_features = X.shape
    C = np.vstack([np.eye(n_features), -np.eye(n_features)])
    d = np.hstack([np.ones(n_features), np.zeros(n_features)])
    if qnorm == 2:
        pnorm = 2
    if qnorm == 1:
        pnorm = np.inf
    if np.isinf(qnorm):
        pnorm = 1
    
    # Step 1: Solve DRO
    model = grb.Model('RobustEstimation')
    model.setParam('OutputFlag', verbose)
    model.setAttr('ModelSense', grb.GRB.MINIMIZE)

    var_lambda = model.addVar(
        vtype=grb.GRB.CONTINUOUS,
        obj=epsilon,
        name='lambda'
        )
    var_s = model.addVars(
        n_samples,
        vtype=grb.GRB.CONTINUOUS,
        obj=1.0/n_samples,
        name='s'
        )
    var_theta = model.addVars(
        n_features,
        vtype=grb.GRB.CONTINUOUS,
        lb=-grb.GRB.INFINITY,
        name='theta'
        )
    var_p = model.addVars(
        n_samples, d.size,
        vtype=grb.GRB.CONTINUOUS,
        name='p'
        )

    model.addConstrs(
        (1 - y[i] * grb.quicksum(var_theta[j] * X[i, j] for j in range(n_features)) +
         grb.quicksum(var_p[i, k] * d[k]
                      for k in range(d.size)) -
         grb.quicksum(var_p[i, k] * C[k, j] * X[i, j]
                      for j in range(n_features)
                      for k in range(d.size)) <= var_s[i]
         for i in range(n_samples))
        )

    if pnorm == 1:
        slack_var_1 = model.addVars(
            n_samples, n_features,
            vtype=grb.GRB.CONTINUOUS
            )
        model.addConstrs(
            (grb.quicksum(var_p[i, k] * C[k, j] for k in range(d.size)) +
             y[i] * var_theta[j] <= slack_var_1[i, j]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstrs(
            (grb.quicksum(-var_p[i, k] * C[k, j] for k in range(d.size)) -
             y[i] * var_theta[j] <= slack_var_1[i, j]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstrs(
            (grb.quicksum(slack_var_1[i, j] for j in range(n_features)) <= var_lambda
             for i in range(n_samples))
        )

    elif pnorm == 2:
        for i in range(n_samples):
            model.addQConstr(
                grb.quicksum(
                    var_theta[j] * var_theta[j]
                    for j in range(n_features)) +
                grb.quicksum(
                    var_p[i, k1] * C[k1, j] * var_p[i, k2] * C[k2, j]
                    for j in range(n_features)
                    for k1 in range(d.size)
                    for k2 in range(d.size)) +
                grb.quicksum(
                    2 * var_p[i, k] * C[k, j] * y[i] * var_theta[j]
                    for j in range(n_features)
                    for k in range(d.size)) <= var_lambda * var_lambda)

    elif np.isinf(pnorm):
        model.addConstrs(
            (grb.quicksum(var_p[i, k] * C[k, j] for k in range(d.size)) +
             y[i] * var_theta[j] <= var_lambda
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstrs(
            (grb.quicksum(-var_p[i, k] * C[k, j] for k in range(d.size)) -
             y[i] * var_theta[j] <= var_lambda
             for i in range(n_samples)
             for j in range(n_features))
        )

    model.optimize()

    theta = np.array([value.x for value in var_theta.values()])
    
    
    # Step 2: solve the worst-case problem
    model = grb.Model('WorstCase')
    model.setParam('OutputFlag', verbose)

    var_q = model.addVars(
        n_samples,
        vtype=grb.GRB.CONTINUOUS,
        lb=0,
        ub=1,
        name='q'
    )
    var_xi = model.addVars(
        n_samples, n_features,
        vtype=grb.GRB.CONTINUOUS,
        lb=-grb.GRB.INFINITY,
        name='xi'
    )

    model.addConstrs(
        (var_q[i] * X[i, j] - var_xi[i, j] >= 0)
        for i in range(n_samples)
        for j in range(n_features)
    )
    model.addConstrs(
        (var_q[i] * X[i, j] - var_xi[i, j] <= var_q[i])
        for i in range(n_samples)
        for j in range(n_features)
    )

    if qnorm == 1:
        slack_var = model.addVars(
            n_samples, n_features,
            vtype=grb.GRB.CONTINUOUS
        )
        model.addConstrs(
            (var_xi[i, j] <= slack_var[i, j]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstrs(
            (-var_xi[i, j] <= slack_var[i, j]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstr(
            grb.quicksum(slack_var[i, j]
                         for i in range(n_samples)
                         for j in range(n_features)) <= n_samples * epsilon
        )

    elif qnorm == 2:
        slack_var = model.addVars(
            n_samples,
            vtype=grb.GRB.CONTINUOUS
        )
        for i in range(n_samples):
            model.addQConstr(
                grb.quicksum(var_xi[i, j] * var_xi[i, j]
                             for j in range(n_features)) <= slack_var[i] * slack_var[i]
            )
        model.addConstr(
            grb.quicksum(slack_var[i]
                         for i in range(n_samples)) <= n_samples * epsilon
        )

    elif np.isinf(qnorm):
        slack_var = model.addVars(
            n_samples,
            vtype=grb.GRB.CONTINUOUS
        )
        model.addConstrs(
            (var_xi[i, j] <= slack_var[i]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstrs(
            (-var_xi[i, j] <= slack_var[i]
             for i in range(n_samples)
             for j in range(n_features))
        )
        model.addConstr(
            grb.quicksum(slack_var[i]
                         for i in range(n_samples)) <= n_samples * epsilon
        )

    obj = grb.quicksum(var_q[i] for i in range(n_samples)) - \
          grb.quicksum(var_q[i] * y[i] * X[i, j] * theta[j] for i in range(n_samples) for j in range(n_features)) + \
          grb.quicksum(var_xi[i, j] * theta[j] for i in range(n_samples) for j in range(n_features))
    
    model.setObjective(obj / n_samples, grb.GRB.MAXIMIZE)
     
    model.optimize()
    q = np.array([value.x for value in var_q.values()])
    xi = np.array([value.x for value in var_xi.values()])
    xi = np.reshape(xi, (n_samples, n_features))

    return xi, q


def make_distribution(xi, q, tol=1e-6):
    n_samples, n_features = xi.shape
    q[np.linalg.norm(xi, axis=1) < tol] = 0
    xi[q == 0] = np.zeros(n_features)
    return xi, q


def main():
    idx = 400
    n_samples = 10
    DIGITS = [3, 8]
    n_class = int(n_samples / 2)
    n_features = 784
    Z1 = load_svmlight_file('./dataset/MNIST_train_' + str(DIGITS[0]) + '.txt',
                            n_features=n_features)
    Z2 = load_svmlight_file('./dataset/MNIST_train_' + str(DIGITS[1]) + '.txt',
                            n_features=n_features)
    X = vstack([Z1[0][idx:idx+n_class, :], Z2[0][idx:idx+n_class, :]]).toarray() / 255
    y = np.hstack([Z1[1][idx:idx+n_class], Z2[1][idx:idx+n_class]])
    y[y == DIGITS[0]] = -1
    y[y == DIGITS[1]] = 1
    epsilons = np.array([0.01, 0.1, 1, 10])
    norm = 1
    least_favorable_distributions = {}
    worst_case_distributions = {}
    for epsilon in epsilons:
        xi, q = least_favorable_distribution(X, y, epsilon, norm)
        [xi, q] = make_distribution(xi, q) 
        least_favorable_distributions[(norm, epsilon)] = (xi, q)
        
        xi, q = worst_case_distribution(X, y, epsilon, norm)
        [xi, q] = make_distribution(xi, q)
        worst_case_distributions[(norm, epsilon)] = (xi, q)

    fig, arr_ax = gen_image(X, least_favorable_distributions, np.insert(epsilons, 0, 0), np.inf)
    fig.savefig('least_inf.eps', format='eps', bbox_inches='tight', pad_inches = 0.2)

    fig, arr_ax = gen_image(X, worst_case_distributions, np.insert(epsilons, 0, 0), np.inf)
    fig.savefig('worst_inf.eps', format='eps', bbox_inches='tight', pad_inches = 0.2)

if __name__ == "__main__":
    main()