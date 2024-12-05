import numpy as np

def simplex_method(c, A, b, maximize=True):
    """
    Simplex method for solving linear programming problems.
    Handles both maximization and minimization problems.

    Parameters:
        c: Coefficients of the objective function (1D array).
        A: Coefficients of the constraints (2D array).
        b: RHS of the constraints (1D array).
        maximize: True for maximization, False for minimization.

    Returns:
        Optimal solution and objective value (maximized/minimized).
    """
    if not maximize:
        c = -c  # Convert minimization to maximization

    # Number of constraints and variables
    num_constraints, num_variables = A.shape

    # Add slack variables to convert inequalities to equalities
    slack = np.eye(num_constraints)
    tableau = np.hstack((A, slack, b.reshape(-1, 1)))

    # Add objective function row
    z_row = np.hstack((-c, np.zeros(num_constraints + 1)))
    tableau = np.vstack((tableau, z_row))

    # Simplex iterations
    while True:
        # Step 1: Optimality test
        if all(tableau[-1, :-1] >= 0):
            break

        # Step 2: Identify entering variable (most negative in Z-row)
        pivot_col = np.argmin(tableau[-1, :-1])

        # Step 3: Identify leaving variable (minimum ratio test)
        ratios = tableau[:-1, -1] / tableau[:-1, pivot_col]
        ratios[ratios <= 0] = np.inf  # Ignore non-positive ratios
        pivot_row = np.argmin(ratios)

        # Step 4: Pivot operation
        pivot_value = tableau[pivot_row, pivot_col]
        tableau[pivot_row, :] /= pivot_value
        for i in range(len(tableau)):
            if i != pivot_row:
                tableau[i, :] -= tableau[i, pivot_col] * tableau[pivot_row, :]

    # Extract solution
    solution = np.zeros(num_variables)
    for i in range(num_variables):
        col = tableau[:-1, i]
        if np.count_nonzero(col) == 1 and np.sum(col) == 1:
            row = np.argmax(col)
            solution[i] = tableau[row, -1]

    # Maximum value of Z
    max_value = tableau[-1, -1]

    return solution, max_value
