from dataclasses import asdict
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from matplotlib import rc
from sklearn.covariance import LedoitWolf

rc('font', family='serif')
rc('text', usetex=True)


def plot_with_shade(x, y, xlabel, ylabel):
    y_mean = np.mean(y, axis=0)
    y_max = np.max(y, axis=0)
    y_min = np.min(y, axis=0)
    fig, ax = plt.subplots(1)
    ax.plot(x, y_mean, lw=2, color=[0, 0.4470, 0.7410])
    ax.set_xlabel(xlabel, fontsize=18)
    ax.set_ylabel(ylabel, fontsize=18, labelpad=-20)
    ax.set_xlim(x.min(), x.max())
    ax.set_yticks((-0.0358,-0.0345))
    ax.set_xscale('log')
    ax.ticklabel_format(axis='y', style='sci', scilimits=(-2,-2))
    fig.savefig('spot.pdf')  
    return fig, ax

def Ind_retunrs(N, is_train):
    fname = '10_Industry_Portfolios.CSV'
    df_10ind = pd.read_csv(fname, skiprows=11, nrows=1123, index_col=0)
    if is_train:
        data = np.log(100 + df_10ind.loc['197001':'202001'].values) - np.log(100)
        mu = np.mean(data, axis=0)
        #cov = np.cov(data.T)
        cov = LedoitWolf().fit(data).covariance_
        R = np.exp(np.random.multivariate_normal(mu, cov, N))
    else:
        data = np.log(100 + df_10ind.loc['199001':'202001'].values) - np.log(100)
        mu = np.mean(data, axis=0)
        cov = np.cov(data.T)
        R = np.exp(np.random.multivariate_normal(mu, cov, N))
    return R

def log_obj_grad(theta, xi, eps_tr):
    scores = np.einsum('i,ji->j', theta, xi)
    obj = np.mean(-np.log(scores + eps_tr))
    grad = np.einsum('ji,j->ji', -xi, 1/ (scores + eps_tr))
    return obj, np.mean(grad, axis=0)

def robust_obj_grad(x, xi, epsilon, eps_tr):
    if xi.ndim == 1:
        xi = xi[np.newaxis, :]
    m, n = np.shape(xi)
    theta = x[0:n]
    lambdaa = x[-1]

    # Sort & unsort indices
    scores = np.einsum('i,ji->ji', theta, xi)
    sorted_indices = np.argsort(scores, axis=1)
    sorted_reverse = np.argsort(sorted_indices, axis=1)
    idx = np.ogrid[:m, :n]
    idx_r = np.ogrid[:m, :n]
    idx[1] = sorted_indices
    idx_r[1] = sorted_reverse

    # Compute alpha_star
    sorted_scores = scores[tuple(idx)]
    alpha = np.cumsum(sorted_scores, axis=1)[:, ::-1]
    alpha = np.einsum('i,ji->ji', 1 - np.arange(n) * lambdaa, 1 / (alpha + eps_tr))
    thresholds = lambdaa / (sorted_scores[:, ::-1] + eps_tr)
    ind = alpha <= thresholds
    k_star = n - np.sum(ind, axis=1)
    alpha_star = np.choose(k_star, alpha.T)

    # Compute objective_star
    temp = np.einsum('j,ji->ji', alpha_star, scores)
    ind2 = ind[:, ::-1]
    ind2 = ind2[tuple(idx_r)]
    all_obj = (temp * ind2) + (lambdaa + lambdaa * np.log(temp / lambdaa + eps_tr)) * ~ind2
    obj_star = lambdaa * epsilon + 1 + np.log(alpha_star) - np.sum(all_obj, axis=1)

    # Compute gradients
    temp2 = np.zeros(temp.shape)
    temp2[~ind2] = -np.log(temp[~ind2] / lambdaa + eps_tr)
    grad_lambdaa = np.sum(temp2, axis=1)
    grad_theta = np.einsum('j,ji->ji', -alpha_star, xi)
    grad_theta[~ind2] = - lambdaa / (np.tile(theta, (m, 1))[~ind2] + eps_tr)
    return np.mean(obj_star), np.append(np.mean(grad_theta, axis=0),
                               epsilon + np.mean(grad_lambdaa))

def projection_onto_simplex(theta):
    n_features = theta.shape[0]
    u = np.sort(theta)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    treshold = cssv[cond][-1] / float(rho)
    return np.maximum(theta - treshold, 0.0)

def dro(xi, epsilon, x_0=None, eps_tr=1e-15, iter_max=1e6):
    n_features = xi.shape[1]
    x_current = x_0
    if x_0 is None:
        x_current = np.append(np.ones(n_features) / n_features, 2 / n_features)
    x_prev = np.random.uniform(0, 1, n_features)
    x_prev = np.append(x_prev / np.sum(x_prev), 1.5 / n_features)
    x_bar = x_current.copy()
    phi = 1.5
    rho = 1.0 / phi + 1.0 / (phi**2)
    tau = 1
    eta_bar = 1e6
    grad_current = robust_obj_grad(x_current, xi, epsilon, eps_tr)[1]
    grad_prev = robust_obj_grad(x_prev, xi, epsilon, eps_tr)[1]
    eta = np.linalg.norm(x_current - x_prev) / \
          np.linalg.norm(grad_current - grad_prev)
    obj = []
    for iter in range(int(iter_max)):
        obj_current, grad_current = robust_obj_grad(x_current, xi, epsilon, eps_tr)
        grad_prev = robust_obj_grad(x_prev, xi, epsilon, eps_tr)[1]
        obj.append(obj_current)
        temp = phi * tau * np.linalg.norm(x_current - x_prev)** 2 / \
               (4 * eta * np.linalg.norm(grad_current - grad_prev)** 2 + eps_tr)
        eta = np.min([rho * eta, temp, eta_bar])
        x_bar = (1 - 1.0 / phi) * x_current + x_bar / phi
        x = x_bar - eta * grad_current
        x[0:n_features] = projection_onto_simplex(x[0:n_features])
        if x[-1] <= 1.0 / n_features:
            x[-1] = 1.0000001 / n_features
        x_prev = x_current.copy()
        x_current = x.copy()
        if iter > 0 and iter % 100 == 0:
            imp = np.abs(np.diff(np.array(obj)[-100:]))
            if np.mean(imp) < eps_tr:
                break
    theta = x[0:n_features]
    lambdaa = x[-1]
    return theta, lambdaa, obj

def saa(xi, eps_tr=1e-15, iter_max=1e6):
    n_features = xi.shape[1]
    theta_current = np.ones(n_features) / n_features
    theta_prev = np.random.uniform(0, 1, n_features)
    theta_prev /= np.sum(theta_prev)
    theta_bar = theta_current.copy()
    phi = (np.sqrt(5) + 1) / 2
    rho = 1.0 / phi + 1.0 / (phi**2)
    tau = 1
    eta_bar = 1e6
    grad_current = log_obj_grad(theta_current, xi, eps_tr)[1]
    grad_prev = log_obj_grad(theta_prev, xi, eps_tr)[1]
    eta = np.linalg.norm(theta_current - theta_prev) / \
          np.linalg.norm(grad_current - grad_prev)
    obj = []
    for iter in range(int(iter_max)):
        obj_current, grad_current = log_obj_grad(theta_current, xi, eps_tr)
        grad_prev = log_obj_grad(theta_prev, xi, eps_tr)[1]
        obj.append(obj_current)
        temp = phi * tau * np.linalg.norm(theta_current - theta_prev)** 2 / \
               (4 * eta * np.linalg.norm(grad_current - grad_prev)** 2 + eps_tr)
        eta = np.min([rho * eta, temp, eta_bar])
        theta_bar = (1 - 1.0 / phi) * theta_current + theta_bar / phi
        theta = projection_onto_simplex(theta_bar - eta * grad_current)
        theta_prev = theta_current.copy()
        theta_current = theta.copy()
        if iter > 0 and iter % 100 == 0:
            imp = np.abs(np.diff(np.array(obj)[-100:]) / obj[-1])
            if np.mean(imp) < eps_tr:
                break
    return theta, obj

def main():
    np.random.seed(1000)
    eps_tr = 1e-10
    epsilon_range = np.append(np.concatenate(
        [np.arange(1, 10) * 10.0 ** (i) for i in range(-4, -1)]), 1e-1)
    is_load = False
    if is_load and \
        os.path.isfile('./save/profit_SAA.npy') and \
            os.path.isfile('./save/profit_DRO.npy') and \
                os.path.isfile('./save/cost_SAA.npy') and \
                    os.path.isfile('./save/cost_DRO.npy'):
        print('Load files ...')
        profit_SAA = np.load('./save/profit_SAA.npy')
        profit_DRO = np.load('./save/profit_DRO.npy')
        cost_SAA = np.load('./save/cost_SAA.npy')
        cost_DRO = np.load('./save/cost_DRO.npy')
    else:
        profit_SAA = []
        profit_DRO = []
        cost_SAA = []
        cost_DRO = []
        for r in range(1000):
            print('iteration ', r+1)
            N_train = int(1e2)
            N_test = int(1e6)
            xi_train = Ind_retunrs(N_train, True)
            xi_test = Ind_retunrs(N_test, True)

            theta_1, obj_1 = saa(xi_train, eps_tr)
            cost_SAA.append(-np.mean(np.log(xi_test @ theta_1)))
            profit_SAA.append(np.mean(xi_test @ theta_1))
            cost_tmp = []
            profit_tmp = []
            x_0 = None
            for eps in epsilon_range:
                theta_2, lambdaa_2, obj_2 = dro(xi_train, eps, x_0, eps_tr)
                x_0 = np.append(theta_2, lambdaa_2)
                cost_tmp.append(-np.mean(np.log(xi_test @ theta_2)))
                profit_tmp.append(np.mean(xi_test @ theta_2))
            cost_DRO.append(cost_tmp)
            profit_DRO.append(profit_tmp)
        profit_SAA = np.array(profit_SAA)
        profit_DRO = np.array(profit_DRO)
        cost_SAA = np.array(cost_SAA)
        cost_DRO = np.array(cost_DRO)
        if not os.path.isdir('./save/'):
            os.mkdir('./save/')
        np.save('./save/profit_SAA', profit_SAA)
        np.save('./save/profit_DRO', profit_DRO)
        np.save('./save/cost_SAA', cost_SAA)
        np.save('./save/cost_DRO', cost_DRO)

    bins = 30
    plt.ticklabel_format(axis='x', style='sci', scilimits=(-2,-2))
    plt.hist(cost_SAA, bins, histtype ='bar', alpha=0.4, edgecolor='black', label=r'SAA $(\varepsilon = 0)$', color='red')
    plt.hist(cost_DRO[:,19], bins, histtype ='bar', alpha=0.4, edgecolor='black', label=r'DRO $(\varepsilon = 10^{-2})$', color='blue')
    plt.legend(loc='upper right')
    plt.savefig('hist.pdf') 

    epsilon_range = np.append(np.concatenate(
        [np.arange(1, 10) * 10.0 ** (i) for i in range(-4, -1)]), 1e-1)
    plot_with_shade(epsilon_range, cost_DRO, 
                    r'$\varepsilon$', 
                    r'$ \mathbb E_{Z \sim \mathbb P}[-\log(\theta^\top Z )]$')


if __name__ == "__main__":
    main()