import torch
from . import util


def pnqp(H, q, lower, upper, x_init=None, n_iter=20):
    """
    Implements a Projected Newton Quadratic Programming (PNQP) algorithm.

    Parameters:
    - H: The Hessian matrix.
    - q: The gradient vector.
    - lower: Lower bounds for the variables.
    - upper: Upper bounds for the variables.
    - x_init: Initial solution. If None, the function computes an initial solution.
    - n_iter: Maximum number of iterations.

    Returns:
    - x: Solution vector.
    - H_ or H_lu_: Updated Hessian or its LU factorization.
    - If: Feasibility indicator vector.
    - i: Number of iterations.
    """

    # Regularization term for positive definiteness of the Hessian
    GAMMA = 0.1
    n_batch, n, _ = H.size()
    pnqp_I = 1e-11 * torch.eye(n, dtype=H.dtype, device=H.device).expand_as(H)

    # Objective function definition for the QP
    def obj(x):
        return 0.5 * util.bquad(x, H) + util.bdot(q, x)

    # Compute the initial solution if it's not provided
    if x_init is None:
        if n == 1:
            x_init = -(1. / H.squeeze(2)) * q
        else:
            H_lu = torch.linalg.lu_factor(H)
            x_init = -torch.linalg.lu_solve(*H_lu, q.unsqueeze(2)).squeeze(2)
    else:
        x_init = x_init.clone()  # Don't over-write the original x_init.

    # Initialize x while ensuring it's within the provided bounds
    x = torch.clamp(x_init, lower, upper)

    # Start iterations for the PNQP algorithm
    for i in range(n_iter):
        # Compute the gradient at the current point
        g = util.bmv(H, x) + q

        # Compute indicators for constraints
        Ic = (((x == lower) & (g > 0)) | ((x == upper) & (g < 0)))
        If = 1 - Ic.float()

        # Create mask for feasible updates on Hessian
        Hff_I = util.bger(If, If)
        not_Hff_I = 1 - Hff_I

        # Modify gradient and Hessian based on feasibility
        g_ = g.clone()
        g_[Ic] = 0.
        H_ = H.clone()
        H_[not_Hff_I.bool()] = 0.0
        H_ += pnqp_I

        # Compute direction of update
        if n == 1:
            dx = -(1. / H_.squeeze(2)) * g_
        else:
            H_lu_ = torch.linalg.lu_factor(H_)
            dx = -torch.linalg.lu_solve(*H_lu_, g_.unsqueeze(2)).squeeze(2)

        # Check if the direction norm is below a threshold
        J = torch.norm(dx, 2, 1) >= 1e-4
        m = J.sum().item()  # Number of active examples in the batch.

        # If no active examples, return the current solution
        if m == 0:
            return x, H_ if n == 1 else H_lu_, If, i

        alpha = torch.ones(n_batch, dtype=x.dtype,
                           device=x.device)  # Initialize step size alpha for all samples to be 1
        decay = 0.1  # Define decay rate for alpha
        max_armijo = GAMMA  # Initialize maximum value for the Armijo-Goldstein condition
        count = 0

        # Armijo line search loop
        while max_armijo <= GAMMA and count < 10:
            # Calculate potential next point by taking a step in the direction of dx
            # and ensuring the result is within the bounds [lower, upper]
            maybe_x = torch.clamp(x + torch.diag(alpha).mm(dx), lower, upper)

            # Initialize Armijo-Goldstein condition values for all samples
            armijos = (GAMMA + 1e-6) * torch.ones(n_batch, dtype=x.dtype, device=x.device)

            # Compute the Armijo-Goldstein condition for samples that had a significant update in the last iteration
            armijos[J] = (obj(x) - obj(maybe_x))[J] / util.bdot(g, x - maybe_x)[J]

            I = armijos <= GAMMA  # Identify samples that do not satisfy the Armijo-Goldstein condition
            alpha[I] *= decay  # Decay alpha for those samples
            max_armijo = torch.max(armijos)  # Update the maximum value for the Armijo-Goldstein condition
            count += 1
        x = maybe_x  # Update x to the new potential value after line search

    # TODO: Maybe change this to a warning.
    # print("[WARNING] pnqp warning: Did not converge")
    return x, H_ if n == 1 else H_lu_, If, i
