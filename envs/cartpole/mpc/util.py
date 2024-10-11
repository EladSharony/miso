import torch


def jacobian(f, x, eps):
    """
    Computes the Jacobian of a function f at x using finite differences.
    :param f: function
    :param x: input
    :param eps: epsilon
    :return: Jacobian of f at x
    """
    if x.ndimension() == 2:
        assert x.size(0) == 1
        x = x.squeeze()

    x_high_precision = x.double()
    n = x_high_precision.numel()
    e = torch.eye(n, dtype=torch.float64, device=x_high_precision.device)
    f_original = f(x_high_precision)
    J_separate = []
    for i in range(n):
        perturbed_value = f(x_high_precision + eps * e[i])
        for j, f_val in enumerate(perturbed_value.unbind()):
            J_separate.append((f_val - f_original[j]) / eps)
        # J_separate.append((perturbed_value - f_original) / eps)
    J_high_precision = torch.stack(J_separate).reshape(n, -1).T
    return J_high_precision.to(dtype=x.dtype)


def bdiag(d):
    """
    Constructs a batched diagonal matrix from a batched vector.
    :param d: batched vector
    :return: batched diagonal matrix
    """
    n_batch, n_dim = d.size()
    D = torch.zeros(n_batch, n_dim, n_dim, dtype=d.dtype, device=d.device)
    for i in range(n_dim):
        D[:, i, i] = d[:, i]
    return D


def bger(x, y):
    """
    Constructs a batched outer product of two batched vectors.
    :param x: batched vector
    :param y: batched vector
    :return: batched outer product
    """
    return torch.einsum('bi,bj->bij', x, y)


def bmv(X, y):
    """
    Computes a batched matrix-vector product.
    :param X: batched matrix
    :param y: batched vector
    :return: batched matrix-vector product
    """
    return X.bmm(y.unsqueeze(2)).squeeze(2)


def bquad(x, Q):
    """
    Computes a batched quadratic form.
    :param x: batched vector
    :param Q: batched matrix
    :return: batched quadratic form
    """
    return x.unsqueeze(1).bmm(Q).bmm(x.unsqueeze(2)).squeeze(1).squeeze(1)


def bdot(x, y):
    """
    Computes a batched dot product.
    :param x: batched vector
    :param y: batched vector
    :return: batched dot product
    """
    return torch.bmm(x.unsqueeze(1), y.unsqueeze(2)).squeeze(1).squeeze(1)


def get_data_maybe(x):
    """
    Returns the data of a tensor, or the input itself if it is not a tensor.
    """
    return x.detach() if torch.is_tensor(x) else x


def table_log(headers, values):
    """
    Logs a table with headers and values.

    Args:
    - headers (list): List of headers for the table.
    - values (list of lists): Rows of values for the table.
    """

    # Define a helper function to print a row
    def print_row(values, width=15):
        """Helper function to print a row in the table."""
        print("".join([str(val).ljust(width) for val in values]))

    # Print the table
    separator = ["-" * 15 for _ in headers]
    print_row(separator)
    print_row(headers)
    print_row(separator)
    for value_row in values:
        print_row(value_row)


def get_traj(T, u, x_init, dynamics):
    """
    Returns a trajectory given a control sequence and initial state.

    Args:
    - T (int): Number of time steps.
    - u (list of tensors): List of control inputs.
    - x_init (tensor): Initial state.
    - dynamics (function): Function that computes the next state given the
        current state and control input.

    Returns:
    - x (tensor): Trajectory.
    """
    from .mpc import LinDx

    x = torch.zeros(T, *x_init.shape, dtype=x_init.dtype, device=x_init.device)
    x[0, ...] = x_init
    if isinstance(dynamics, LinDx):
        F, f = get_data_maybe(dynamics.F), get_data_maybe(dynamics.f)
        if f is not None:
            assert f.shape == F.shape[:3]
        for t in range(T - 1):
            xt = x[t]
            ut = get_data_maybe(u[t])
            xut = torch.cat((xt, ut), 1)
            new_x = bmv(F[t], xut)
            if f is not None:
                new_x += f[t]
            x[t+1, ...] = new_x
    else:
        for t in range(T - 1):
            xt = x[t]
            ut = get_data_maybe(u[t])
            new_x = dynamics(xt, ut).detach()
            x[t+1, ...] = new_x
    return x


def get_cost(T, u, cost, dynamics=None, x_init=None, x=None):
    """
    Returns the cost of a trajectory given a control sequence.

    Args:
    - T (int): Number of time steps.
    - u (list of tensors): List of control inputs.
    - cost (function): Function that computes the cost of a state and control
        input.
    - dynamics (function): Function that computes the next state given the
        current state and control input.
    - x_init (tensor): Initial state.
    - x (tensor): Trajectory.

    Returns:
    - total_obj (tensor): Total cost of the trajectory.
    """
    from .mpc import QuadCost

    assert x_init is not None or x is not None
    if x is None:
        x = get_traj(T, u, x_init, dynamics)

    if isinstance(cost, QuadCost):
        C, c = get_data_maybe(cost.C), get_data_maybe(cost.c)
        xut = torch.cat((x[0], u[0]), 1)
        obj = 0.5 * bquad(xut, C[0]) + bdot(xut, c[0])
        total_obj = obj
        for t in range(T-1):
            xut = torch.cat((x[t+1], u[t+1]), 1)
            obj = 0.5*bquad(xut, C[t+1]) + bdot(xut, c[t+1])
            total_obj += obj
    else:
        xut = torch.cat((x[0], u[0]), 1)
        obj = cost(xut)
        total_obj = obj
        for t in range(T-1):
            xut = torch.cat((x[t+1], u[t+1]), 1)
            obj = cost(xut)
            total_obj += obj
    return total_obj


def detach_maybe(x):
    """
    Detaches a tensor if it requires grad.
    """
    if x is None:
        return None
    return x if not x.requires_grad else x.detach()
