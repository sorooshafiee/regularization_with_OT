import numpy as np
import gurobipy as grb
import matplotlib.pyplot as plt
from matplotlib import rc
from matplotlib.colors import ListedColormap
from sklearn.datasets import make_blobs

rc("font", family="serif")
rc("text", usetex=True)
rc("xtick", labelsize=15)
rc("ytick", labelsize=15)


def plot_data(X, y, theta, fname, perturbation, weights):
    # Set the borders and grids
    x = 2.5
    x1_min, x1_max = -x, x
    x2_min, x2_max = -x, x
    XX1, XX2 = np.meshgrid(
        np.linspace(x1_min, x1_max, 100), np.linspace(x2_min, x2_max, 100)
    )
    XX = np.c_[XX1.ravel(), XX2.ravel()]
    Z = XX @ theta
    Z = Z.reshape(XX1.shape)

    # Settings of figures
    fig, ax = plt.subplots(1, 3, figsize=[12, 3.5])
    cm = ListedColormap(["#BA1B1D", "#082678"])
    alpha = 1
    s = 100

    # Define markers and arrow properties
    markers = {1: "+", -1: "_"}
    arrow_properties = dict(
        arrowstyle="->",
        color="#000000",
        lw=2,
    )

    # Identify support vectors (SVs)
    ind = weights != 0
    indeces = np.where(ind)[0]
    sv_X = X[ind]
    sv_y = y[ind]
    nsv_X = X[~ind]
    nsv_y = y[~ind]

    # Plot data points in all three subplots
    for i in range(3):
        ax[i].set_xlim(x1_min, x1_max)
        ax[i].set_ylim(x2_min, x2_max)

        ax[i].scatter(
            nsv_X[nsv_y == 1, 0],
            nsv_X[nsv_y == 1, 1],
            c=[cm.colors[1]],
            s=s,
            marker=markers[1],
            alpha=alpha,
            linewidths=3,
        )

        ax[i].scatter(
            nsv_X[nsv_y == -1, 0],
            nsv_X[nsv_y == -1, 1],
            c=[cm.colors[0]],
            s=s,
            marker=markers[-1],
            alpha=alpha,
            linewidths=3,
        )

        ax[i].contour(
            XX1,
            XX2,
            Z,
            colors="#999999",
            levels=[-1, 0, 1],
            linestyles=["--", "-", "--"],
            zorder=0,
        )
        ax[i].set_xticks((-2, 2))
        ax[i].set_yticks((-2, 2))
        ax[i].set_xlabel(r"$x_1$", fontsize=20, labelpad=-10)
        ax[i].set_ylabel(r"$x_2$", fontsize=20, labelpad=-20)

    # --- Left Plot: Circle Support Vectors ---
    for cls in [-1, 1]:
        class_mask = sv_y == cls
        if np.any(class_mask):
            color = cm.colors[1] if cls == 1 else cm.colors[0]

            ax[0].scatter(
                sv_X[class_mask, 0],
                sv_X[class_mask, 1],
                s=s,
                c=[color],
                marker=markers[cls],
                alpha=1,
                linewidths=2,
            )

            ax[0].scatter(
                sv_X[class_mask, 0],
                sv_X[class_mask, 1],
                s=s + 80,
                facecolors="none",
                edgecolors=[color],
                linewidths=2,
            )

    # --- Middle Plot: Single Perturbation with Arrow ---
    index = indeces[-1]
    X_orig = X[index]
    y_orig = y[index]
    X_p = X_orig - y_orig * perturbation
    color = [cm.colors[1] if y_orig == 1 else cm.colors[0]]
    marker = markers[y_orig]

    ax[1].scatter(
        X_orig[0], X_orig[1], s=s, c=color, marker=marker, alpha=alpha / 2, linewidths=2
    )
    ax[1].scatter(X_p[0], X_p[1], s=s, c=color, marker=marker, alpha=1, linewidths=2)
    ax[1].annotate("", xy=X_p, xytext=X_orig, arrowprops=arrow_properties)

    for cls in [-1, 1]:
        class_mask = sv_y == cls
        class_mask[-1] = False
        if np.any(class_mask):
            color = cm.colors[1] if cls == 1 else cm.colors[0]

            ax[1].scatter(
                sv_X[class_mask, 0],
                sv_X[class_mask, 1],
                s=s,
                c=[color],
                marker=markers[cls],
                alpha=1,
                linewidths=2,
            )

    # --- Right Plot: Multiple Perturbations with Arrows ---
    pert_scaled = perturbation / len(indeces)
    Xs_p = sv_X - np.einsum("i,j->ij", sv_y, pert_scaled)

    # Plot original and perturbed SVs as solid
    for cls in [-1, 1]:
        class_mask = sv_y == cls
        if np.any(class_mask):
            color = [cm.colors[1] if cls == 1 else cm.colors[0]]
            ax[2].scatter(
                sv_X[class_mask, 0],
                sv_X[class_mask, 1],
                s=s,
                c=color,
                marker=markers[cls],
                alpha=alpha / 2,
                linewidths=2,
            )
            ax[2].scatter(
                Xs_p[class_mask, 0],
                Xs_p[class_mask, 1],
                s=s,
                c=color,
                marker=markers[cls],
                alpha=1,
                linewidths=2,
            )

    # Add arrows from original to perturbed positions
    for i in range(len(sv_X)):
        original_pos = sv_X[i]
        perturbed_pos = Xs_p[i]
        ax[2].annotate(
            "", xy=perturbed_pos, xytext=original_pos, arrowprops=arrow_properties
        )

    # Save the plot
    fig.savefig(fname, format="pdf", dpi=300, bbox_inches="tight")


def estimation_problem(X, y, epsilon=0.0, norm=1, verbose=False):
    n_samples, n_features = X.shape
    # Step 0: create model for minimization
    model = grb.Model("Estimation")
    model.setParam("OutputFlag", verbose)
    model.setAttr("ModelSense", grb.GRB.MINIMIZE)

    # Step 1: define decision variables & their coefficients in objective function
    var_r = model.addVars(
        n_samples, vtype=grb.GRB.CONTINUOUS, obj=1.0 / n_samples, name="r"
    )
    var_theta = model.addVars(
        n_features, vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, name="theta"
    )
    var_lambda = model.addVar(vtype=grb.GRB.CONTINUOUS, obj=epsilon, name="lambda")

    # Step 2: define constraints
    model.addConstrs(
        (
            1 - y[i] * grb.quicksum(var_theta[j] * X[i, j] for j in range(n_features))
            <= var_r[i]
            for i in range(n_samples)
        )
    )

    if norm == 1:
        slack_var = model.addVars(n_features, vtype=grb.GRB.CONTINUOUS)
        model.addConstrs((var_theta[j] <= slack_var[j] for j in range(n_features)))
        model.addConstrs((-var_theta[j] <= slack_var[j] for j in range(n_features)))
        model.addConstr(
            grb.quicksum(slack_var[j] for j in range(n_features)) <= var_lambda
        )

    elif norm == 2:
        model.addQConstr(
            grb.quicksum(var_theta[j] * var_theta[j] for j in range(n_features))
            <= var_lambda * var_lambda
        )

    elif np.isinf(norm):
        model.addConstrs((var_theta[j] <= var_lambda for j in range(n_features)))
        model.addConstrs((-var_theta[j] <= var_lambda for j in range(n_features)))

    # Step 3: now optimize
    model.optimize()
    theta = np.array([value.x for value in var_theta.values()])

    return theta


def least_favorable_problem(X, y, epsilon=0.0, norm=1, verbose=False):
    n_samples, n_features = X.shape
    # Step 0: create model for minimization
    model = grb.Model("LeastFavorable")
    model.setParam("OutputFlag", verbose)
    model.setAttr("ModelSense", grb.GRB.MAXIMIZE)

    # Step 1: define decision variables & their coefficients in objective function
    var_q = model.addVars(
        n_samples, vtype=grb.GRB.CONTINUOUS, obj=1 / n_samples, lb=0, ub=1, name="q"
    )
    var_xi = model.addVars(
        n_samples, n_features, vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, name="xi"
    )

    # Step 2: define constraints
    model.addConstrs(
        (
            grb.quicksum(var_xi[i, j] for i in range(n_samples))
            == grb.quicksum(var_q[i] * y[i] * X[i, j] for i in range(n_samples))
            for j in range(n_features)
        )
    )

    if norm == 1:
        slack_var = model.addVars(n_samples, n_features, vtype=grb.GRB.CONTINUOUS)
        model.addConstrs(
            (
                var_xi[i, j] <= slack_var[i, j]
                for i in range(n_samples)
                for j in range(n_features)
            )
        )
        model.addConstrs(
            (
                -var_xi[i, j] <= slack_var[i, j]
                for i in range(n_samples)
                for j in range(n_features)
            )
        )
        model.addConstr(
            grb.quicksum(
                slack_var[i, j] for i in range(n_samples) for j in range(n_features)
            )
            <= n_samples * epsilon
        )

    elif norm == 2:
        slack_var = model.addVars(n_samples, vtype=grb.GRB.CONTINUOUS)
        for i in range(n_samples):
            model.addQConstr(
                grb.quicksum(var_xi[i, j] * var_xi[i, j] for j in range(n_features))
                <= slack_var[i] * slack_var[i]
            )
        model.addConstr(
            grb.quicksum(slack_var[i] for i in range(n_samples)) <= n_samples * epsilon
        )

    elif np.isinf(norm):
        slack_var = model.addVars(n_samples, vtype=grb.GRB.CONTINUOUS)
        model.addConstrs(
            (
                var_xi[i, j] <= slack_var[i]
                for i in range(n_samples)
                for j in range(n_features)
            )
        )
        model.addConstrs(
            (
                -var_xi[i, j] <= slack_var[i]
                for i in range(n_samples)
                for j in range(n_features)
            )
        )
        model.addConstr(
            grb.quicksum(slack_var[i] for i in range(n_samples)) <= n_samples * epsilon
        )

    # Step 3: now optimize
    model.optimize()
    q = np.array([value.x for value in var_q.values()])
    xi = np.array([value.x for value in var_xi.values()])
    q[abs(q) < 1e-3] = 0
    q[q > 0.999] = 1

    return np.reshape(xi, (n_samples, n_features)), y, q


def least_favorable_problem_light(X, y, epsilon=0.0, norm=1, verbose=False):
    n_samples, n_features = X.shape
    # Step 0: create model for minimization
    model = grb.Model("LeastFavorable")
    model.setParam("OutputFlag", verbose)
    model.setAttr("ModelSense", grb.GRB.MAXIMIZE)

    # Step 1: define decision variables & their coefficients in objective function
    var_q = model.addVars(
        n_samples, vtype=grb.GRB.CONTINUOUS, obj=1 / n_samples, lb=0, ub=1, name="q"
    )
    var_xi = model.addVars(
        n_features, vtype=grb.GRB.CONTINUOUS, lb=-grb.GRB.INFINITY, name="xi"
    )

    # Step 2: define constraints
    model.addConstrs(
        (
            var_xi[j]
            == grb.quicksum(var_q[i] * y[i] * X[i, j] for i in range(n_samples))
            for j in range(n_features)
        )
    )

    if norm == 1:
        slack_var = model.addVars(n_features, vtype=grb.GRB.CONTINUOUS)
        model.addConstrs((var_xi[j] <= slack_var[j] for j in range(n_features)))
        model.addConstrs((-var_xi[j] <= slack_var[j] for j in range(n_features)))
        model.addConstr(
            grb.quicksum(slack_var[j] for j in range(n_features)) <= n_samples * epsilon
        )

    elif norm == 2:
        slack_var = model.addVar(vtype=grb.GRB.CONTINUOUS)
        model.addQConstr(
            grb.quicksum(var_xi[j] * var_xi[j] for j in range(n_features))
            <= slack_var * slack_var
        )
        model.addConstr(slack_var <= n_samples * epsilon)

    elif np.isinf(norm):
        slack_var = model.addVar(vtype=grb.GRB.CONTINUOUS)
        model.addConstrs((var_xi[j] <= slack_var for j in range(n_features)))
        model.addConstrs((-var_xi[j] <= slack_var for j in range(n_features)))
        model.addConstr(slack_var <= n_samples * epsilon)

    # Step 3: now optimize
    model.optimize()
    q = np.array([value.x for value in var_q.values()])
    xi = np.array([value.x for value in var_xi.values()])
    q[abs(q) < 1e-3] = 0
    q[q > 0.999] = 1

    return xi, q


def main():
    # Set Parameters
    norm = 1
    epsilon = 0.1
    n_samples = 20
    std = 0.5
    centers = np.array([[1, -1], [-1, 1]])

    # Create dataset
    X, y = make_blobs(
        n_samples=n_samples,
        centers=centers,
        cluster_std=std,
        n_features=2,
        random_state=12345,
    )
    y[y == 0] = -1
    X[5] = X[5] + np.array([0.3, 0.3])  # For better visualization

    # Solve the estimation and least-favorable problem
    theta = estimation_problem(X, y, epsilon, norm)
    dual_norm = norm
    if norm == 1:
        dual_norm = np.inf
    if np.isinf(norm):
        dual_norm = 1
    xi, q = least_favorable_problem_light(X, y, epsilon, dual_norm)
    fname = (
        "nash_norm_infty.pdf"
        if np.isinf(dual_norm)
        else "nash_norm_{}.pdf".format(dual_norm)
    )
    plot_data(X, y, theta, fname, xi, q)
    pass


if __name__ == "__main__":
    main()
