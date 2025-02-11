import numpy as np
import math

class LGL:
    """
    Class for computing Legendre–Gauss–Lobatto (LGL) nodes, weights, and 
    the differentiation matrix D based on the Lagrange basis polynomials.

    Attributes:
        p (int): Degree of the Legendre polynomial (the quadrature has p+1 nodes).
        nodes (np.ndarray): The computed LGL nodes on the interval [-1, 1].
        weights (np.ndarray): The computed quadrature weights.
    """
    
    def __init__(self, p):
        """
        Initialize the LGL object with a given polynomial degree p.

        Parameters:
            p (int): Degree of the Legendre polynomial.
                     (Note: There will be p+1 nodes.)
        """
        self.p = p
        self.nodes, self.weights = self._compute_nodes_weights()
    
    @staticmethod
    def legendre_poly_coeffs(p):
        """
        Compute the coefficients of the Legendre polynomial P_p(x) using the formula:
        
            P_p(x) = 1/2^p * sum from k=0 to floor(p/2) of [ (-1)^k * comb(p, k) * comb(2p - 2k, p) * x^(p-2k) ]
        
        The coefficients are returned in descending order (highest power first).

        Parameters:
            p (int): Degree of the Legendre polynomial.
        
        Returns:
            np.ndarray: Array of coefficients.
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
        
        The nodes are given by the endpoints -1 and 1 plus the zeros of the derivative
        of the Legendre polynomial P_p(x). The weights are computed via:
        
            w_i = 2 / (p * (p + 1) * [P_p(x_i)]^2).

        Returns:
            tuple: (nodes, weights) where
                   nodes is a numpy array of the LGL nodes, and
                   weights is a numpy array of the corresponding quadrature weights.
        """
        P = self.legendre_poly(self.p)
        dP = P.deriv()
        # Interior nodes: zeros of the derivative P'_p(x)
        interior_nodes = np.sort(dP.r.real)
        # Include endpoints -1 and 1.
        nodes = np.concatenate(([-1.0], interior_nodes, [1.0]))
        # Compute the weights.
        weights = 2 / (self.p * (self.p + 1) * (P(nodes) ** 2))
        return nodes, weights
    
    def get_nodes(self):
        """
        Return the computed LGL nodes.
        
        Returns:
            np.ndarray: The LGL nodes.
        """
        return self.nodes
    
    def get_weights(self):
        """
        Return the computed quadrature weights.
        
        Returns:
            np.ndarray: The quadrature weights.
        """
        return self.weights

    def differentiation_matrix(self):
        """
        Compute the differentiation matrix D whose (i, j)-th entry is given by
            D[i, j] = dL_j/dx (x_i),
        where L_j(x) is the Lagrange basis polynomial corresponding to node x_j.

        Using the barycentric formulation, we first define the barycentric weights:
            b_j = 1 / (product for all k not equal to j of (x_j - x_k)),
        and then for i not equal to j:
            D[i, j] = (b_j / b_i) / (x_i - x_j),
        with the diagonal entries determined by:
            D[i, i] = - (sum for all j not equal to i of D[i, j]).

        Returns:
            np.ndarray: The differentiation matrix D of shape (N, N), where N = p+1.
        """
        x = self.nodes
        N = len(x)
        D = np.zeros((N, N))
        # Compute barycentric weights b_j.
        b = np.zeros(N)
        for j in range(N):
            # np.delete(x, j) removes the j-th element so that the product is taken for all k not equal to j.
            b[j] = 1.0 / np.prod(x[j] - np.delete(x, j))
        
        # Compute off-diagonal entries.
        for i in range(N):
            for j in range(N):
                if i != j:
                    D[i, j] = (b[j] / b[i]) / (x[i] - x[j])
            # Compute the diagonal entry so that the sum over each row is zero.
            D[i, i] = -np.sum(D[i, :])
        
        return D

def main():
    # Choose p (degree of the Legendre polynomial). There will be p+1 nodes.
    p = 4  # For example, p = 4 (so N = 5 nodes)
    lgl = LGL(p)
    
    np.set_printoptions(suppress=True)
    print("LGL nodes:")
    print(lgl.get_nodes())
    
    print("\nLGL quadrature weights:")
    print(lgl.get_weights())
    
    D = lgl.differentiation_matrix()
    print("\nMatrix D:")
    print(D)
    
if __name__ == '__main__':
    main()
