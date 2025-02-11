import numpy as np
import math

class LGL:
    """
    Class for computing Legendre–Gauss–Lobatto (LGL) nodes and quadrature weights.
    
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
                     (There will be p+1 nodes.)
        """
        self.p = p
        self.nodes, self.weights = self._compute_nodes_weights()
    
    @staticmethod
    def legendre_poly_coeffs(p):
        """
        Compute the coefficients of the Legendre polynomial P_p(x) using the formula:
        
            P_p(x) = (1/2^p) * sum from k=0 to floor(p/2) of [ (-1)^k * comb(p,k) * comb(2p-2k,p) * x^(p-2k) ].
        
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
        
            w_i = 2 / (p(p+1)[P_p(x_i)]^2).
        
        Returns:
            tuple: (nodes, weights) where
                   nodes is a numpy array of the LGL nodes, and
                   weights is a numpy array of the corresponding quadrature weights.
        """
        P_poly = self.legendre_poly(self.p)
        dP = P_poly.deriv()
        # Interior nodes: zeros of the derivative P'_p(x)
        interior_nodes = np.sort(dP.r.real)
        # Include endpoints -1 and 1.
        nodes = np.concatenate(([-1.0], interior_nodes, [1.0]))
        # Compute the weights.
        weights = 2 / (self.p * (self.p + 1) * (P_poly(nodes) ** 2))
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


def compute_matrices_PQD(lgl):
    """
    Using the LGL nodes and weights, compute the matrices P, Q and the differentiation 
    matrix D = P^{-1} Q as in equation (13)-(14):
    
        P = sum_{l=0}^{N-1} L(eta_l;x) L(eta_l;x)^T omega_l,
        Q = sum_{l=0}^{N-1} L(eta_l;x) (dL/dx)(eta_l;x)^T omega_l,
    
    where:
      - eta_l and omega_l (l = 0,...,N-1) are the LGL nodes and quadrature weights,
      - L(eta_l;x) is the column vector of Lagrange basis polynomials relative to the nodes x,
      - (dL/dx)(eta_l;x) is the column vector of their derivatives.
    
    In our case the collocation nodes x are the same as the quadrature nodes, so that
    L_j(x_i) = delta_{ij} (the Kronecker delta). Hence, the matrix L is the identity and the 
    summation produces:
      P = diag(omega_0, omega_1, ..., omega_{N-1}),
      Q_{ij} = (dL_j/dx)(x_i) omega_i.
    
    Then, D = P^{-1} Q recovers the differentiation matrix whose (i,j) entry is 
        dL_j/dx(x_i).
    
    Parameters:
        lgl (LGL): An instance of the LGL class containing nodes and weights.
    
    Returns:
        P, Q, D (tuple of np.ndarray): The matrices P, Q and the differentiation matrix D.
    """
    nodes = lgl.get_nodes()   # Collocation (and quadrature) nodes
    weights = lgl.get_weights() # Quadrature weights
    N = len(nodes)
    
    # --- Step 1. Compute barycentric weights for the nodes ---
    # These are used to evaluate the Lagrange basis functions.
    b = np.zeros(N)
    for j in range(N):
        b[j] = 1.0 / np.prod(nodes[j] - np.delete(nodes, j))
    
    # --- Step 2. Compute the differentiation matrix via the barycentric formula ---
    # That is, for i != j:
    #   D_bary[i,j] = (b[j] / b[i]) / (nodes[i] - nodes[j]),
    # and for the diagonal:
    #   D_bary[i,i] = -sum_{j != i} D_bary[i,j].
    D_bary = np.zeros((N, N))
    for i in range(N):
        for j in range(N):
            if i != j:
                D_bary[i, j] = (b[j] / b[i]) / (nodes[i] - nodes[j])
        D_bary[i, i] = -np.sum(D_bary[i, :])
    
    # --- Step 3. Form the Lagrange basis matrix L ---
    # For an evaluation at the collocation nodes, the Lagrange basis functions satisfy:
    # L_j(nodes_i) = delta_{ij}. Hence, L is the identity matrix.
    Lmat = np.eye(N)
    
    # --- Step 4. Compute matrices P and Q via quadrature ---
    P = np.zeros((N, N))
    Q = np.zeros((N, N))
    for l in range(N):
        # L(eta_l;x) is the column vector of Lagrange basis evaluations at eta_l.
        # Since the evaluation is at a collocation node, it is the l-th unit vector.
        L_eta = Lmat[l, :].reshape(N, 1)
        # (dL/dx)(eta_l;x) is taken from the differentiation matrix.
        dL_eta = D_bary[l, :].reshape(N, 1)
        P += L_eta @ L_eta.T * weights[l]
        Q += L_eta @ dL_eta.T * weights[l]
    
    # --- Step 5. Compute D = P^{-1} Q ---
    P_inv = np.linalg.inv(P)
    D = P_inv @ Q
    return P, Q, D

def main():
    # Choose p (degree of the Legendre polynomial). There will be p+1 nodes.
    p = 4  # For example, p = 4 (so N = 5 nodes)
    lgl = LGL(p)
    
    # Compute matrices P, Q and the differentiation matrix D = P^{-1} Q.
    P, Q, D = compute_matrices_PQD(lgl)
    
    np.set_printoptions(suppress=True)
    print("LGL nodes:")
    print(lgl.get_nodes())
    
    print("\nLGL quadrature weights:")
    print(lgl.get_weights())
    
    print("\nMatrix P:")
    print(P)
    
    print("\nMatrix Q:")
    print(Q)
    
    print("\nMatrix D:")
    print(D)
    
if __name__ == '__main__':
    main()
