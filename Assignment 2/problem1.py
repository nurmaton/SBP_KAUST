import numpy as np
import math

class LGL:
    """
    Class for computing Legendre–Gauss–Lobatto (LGL) nodes, weights, and 
    the differentiation matrix D based on the Lagrange basis polynomials.

    Attributes:
        p (int): Degree of the Legendre polynomial (the quadrature has p+1 nodes).
        L (float): Length of the interval [0, L].
        nodes (np.ndarray): The computed LGL nodes on the interval [-1, 1].
        weights (np.ndarray): The computed quadrature weights.
        D (np.ndarray): The differentiation matrix for computing derivatives.
    """
    
    def __init__(self, p, L=1):
        """
        Initialize the LGL object with a given polynomial degree p and interval length L.

        Parameters:
            p (int): Degree of the Legendre polynomial.
            L (float, optional): Length of the interval [0, L]. Default is 1.
        """
        self.p = p
        self.L = L
        self.nodes, self.weights = self._compute_nodes_weights()
        self.D = self.differentiation_matrix()
    
    @staticmethod
    def legendre_poly_coeffs(p):
        """
        Compute the coefficients of the Legendre polynomial P_p(x).

        Parameters:
            p (int): Degree of the Legendre polynomial.
        
        Returns:
            np.ndarray: Array of coefficients in descending order (highest power first).
        """
        poly_dict = {}
        for k in range(p // 2 + 1):
            power = p - 2 * k
            coeff = ((-1) ** k * math.comb(p, k) * math.comb(2 * p - 2 * k, p)) / (2 ** p)
            poly_dict[power] = coeff
        coeffs = [poly_dict.get(power, 0) for power in range(p, -1, -1)]
        return np.array(coeffs)
    
    @classmethod
    def legendre_poly(cls, p):
        """
        Construct a numpy.poly1d object representing the Legendre polynomial P_p(x).

        Parameters:
            p (int): Degree of the Legendre polynomial.
        
        Returns:
            np.poly1d: The Legendre polynomial.
        """
        coeffs = cls.legendre_poly_coeffs(p)
        return np.poly1d(coeffs)
    
    def _compute_nodes_weights(self):
        """
        Compute the LGL nodes and quadrature weights.
        
        Returns:
            tuple: (nodes, weights) where
                   nodes is a numpy array of the LGL nodes, and
                   weights is a numpy array of the corresponding quadrature weights.
        """
        P = self.legendre_poly(self.p)
        dP = P.deriv()
        interior_nodes = np.sort(dP.r.real)  # Zeros of P'_p(x)
        nodes = np.concatenate(([-1.0], interior_nodes, [1.0]))  # Include endpoints
        weights = 2 / (self.p * (self.p + 1) * (P(nodes) ** 2))
        return nodes, weights
    
    def differentiation_matrix(self):
        """
        Compute the differentiation matrix D.

        Returns:
            np.ndarray: The differentiation matrix D of shape (N, N), where N = p+1.
        """
        x = self.nodes
        N = len(x)
        D = np.zeros((N, N))
        b = np.zeros(N)  # Barycentric weights
        for j in range(N):
            b[j] = 1.0 / np.prod(x[j] - np.delete(x, j))
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (b[j] / b[i]) / (x[i] - x[j])
            D[i, i] = -np.sum(D[i, :])  # Ensure row sum is zero
        return (2 / self.L) * D  # Rescale for [0, L]
    
    def transform_nodes(self):
        """
        Transform LGL nodes from [-1, 1] to [0, L].
        
        Returns:
            np.ndarray: Transformed nodes.
        """
        return (self.L / 2) * (self.nodes + 1)

    def compute_derivative(self, f):
        """
        Compute the derivative of a given function f at the collocation points.

        Parameters:
            f (callable): Function to differentiate.
        
        Returns:
            tuple: (transformed nodes, derivative values at those nodes).
        """
        x_mapped = self.transform_nodes()
        f_values = f(x_mapped)
        return x_mapped, self.D @ f_values

def main():
    p = 4  # Degree of Legendre polynomial
    L = 10  # Interval length
    
    # Define function f(x)
    def f(x):
        return np.full_like(x, 5.0)  # Ensures valid differentiation

    lgl = LGL(p, L)
    y, f_prime = lgl.compute_derivative(f)
    
    print("Mapped nodes y:")
    print(y)
    
    print("\nFunction values f(y):")
    print(f(y))
    
    print("\nComputed derivative df/dy:")
    print(f_prime)
    
if __name__ == '__main__':
    main()
