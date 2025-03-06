import sympy as sp
import numpy as np

class SymbolicLGL:
    """
    Class to compute Legendre-Gauss-Lobatto (LGL) nodes and differentiation matrix D symbolically.
    
    Attributes:
        p (int): Degree of the Legendre polynomial (p+1 nodes).
        nodes (list): Computed LGL nodes as symbolic expressions.
        D (sp.Matrix): Differentiation matrix computed symbolically.
    """
    
    def __init__(self, p):
        """
        Initialize the SymbolicLGL class with polynomial degree p.

        Parameters:
            p (int): Degree of the Legendre polynomial.
        """
        self.p = p
        self.nodes = self.compute_nodes()  # Compute symbolic LGL nodes
        self.D = self.differentiation_matrix()  # Compute symbolic differentiation matrix

    def compute_nodes(self):
        """
        Compute the LGL nodes as symbolic expressions.

        Returns:
            list: LGL nodes including -1 and 1 with symbolic interior nodes.
        """
        x = sp.Symbol('x')  # Define symbolic variable
        P = sp.legendre(self.p, x)  # Legendre polynomial P_p(x)
        dP = sp.diff(P, x)  # Compute derivative P'_p(x)
        
        # Solve for roots of P'_p(x) to get interior nodes
        nodes = sorted(sp.solveset(dP, x, domain=sp.Interval(-1, 1)))
        
        # LGL nodes include -1 and 1
        return [-1] + nodes + [1]

    def differentiation_matrix(self):
        """
        Compute the symbolic differentiation matrix D using the barycentric formula.

        Returns:
            sp.Matrix: Symbolic differentiation matrix.
        """
        x = self.nodes  # LGL nodes
        N = len(x)  # Number of nodes
        D = sp.zeros(N, N)  # Initialize symbolic differentiation matrix

        # Compute barycentric weights: b_j = 1 / (product for all k not equal to j of (x_j - x_k)).
        b = [1 / np.prod([x[j] - x[k] for k in range(N) if k != j]) for j in range(N)]

        # Compute differentiation matrix entries
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (b[j] / b[i]) / (x[i] - x[j])
            D[i, i] = -sum(D[i, :])  # Ensure row sum is zero

        return D

def compute_flux_matrix(nodes):
    """
    Compute the symbolic flux matrix F based on the given LGL nodes.

    The flux function is defined as:
        F_ij = (u_i^2 + u_i u_j + u_j^2) / 6

    Parameters:
        nodes (list): LGL nodes.

    Returns:
        tuple: (F, u) where F is the symbolic flux matrix, and u is the list of symbolic variables.
    """
    N = len(nodes)  # Number of nodes
    u = sp.symbols(f'u1:{N+1}')  # Define symbolic variables u1, u2, ..., uN
    F = sp.zeros(N, N)  # Initialize symbolic flux matrix

    # Compute flux values using the given formula
    for i in range(N):
        for j in range(N):
            F[i, j] = (u[i]**2 + u[i]*u[j] + u[j]**2) / 6

    return F, u

def main():
    """
    Main function to compute and display symbolic and numerical matrices.
    """
    p = 2  # Example polynomial degree
    lgl = SymbolicLGL(p)  # Create LGL object
    
    # Compute differentiation matrix D
    D_symbolic = lgl.D  # Symbolic matrix
    D_numeric = np.array(D_symbolic.evalf().tolist(), dtype=np.float64)  # Numeric conversion

    # Compute flux matrix F
    F_symbolic, u = compute_flux_matrix(lgl.nodes)

    # Print results
    print("\nSymbolic Differentiation Matrix D:")
    sp.pprint(D_symbolic)

    print("\nNumerical Differentiation Matrix D:")
    np.set_printoptions(precision=6, suppress=True)  # Format output
    print(D_numeric)

    print("\nSymbolic Flux Matrix F:")
    sp.pprint(F_symbolic)

if __name__ == '__main__':
    main()
