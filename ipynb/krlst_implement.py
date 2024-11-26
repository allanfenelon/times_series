from sklearn.gaussian_process.kernels import RBF
import numpy as np
import warnings
from typing import Tuple

# REFERENCES: https://www.squobble.com/academic/doc/vanvaerenbergh2012_krls_tracker_tnnls.pdf

class Krls_t:
    def __init__(self, lambda_: float, c: float, M: int, sigma: float = 1.0,
                 kernel_type: str = 'rbf', forgetmode: str = "B2P", nu = 1e-10, using_SVD = False):
        
        self.kernel_type = kernel_type
        if lambda_ < 0 or lambda_ > 1:
            raise ValueError("Parameter `lambda_` is out of allowed range of [0,1].")
        self._lambda = lambda_
        if not (forgetmode in ["B2P"]):
            raise ValueError("Parameter `forgetmode` can either be 'B2P'.")
        self.forgetmode = forgetmode
        self.sigma = sigma
        self.c = c
        self.M = M
        self.nu = nu
        self._kernel = RBF(sigma)
        self._is_init = False
        self.Q = None
        self.mu = None
        self.Sigma = None
        self.basis = None # Dictionary indicies
        self.Xb = None  # Dictionary
        self.m = 0 # Dict size
        self.nums02ML = None
        self.dens02ML = None
        self.U = None
        self.S = None
        self.Vt = None
        self.using_SVD = using_SVD
        self.K_component = False
    
    """Para obter o mínimo de componentes que explique a maior parte da variabilidade dos dados (idealmente próximo a 100%), uma abordagem comum é determinar o menor valor de 
𝑘
k tal que a variância acumulada explicada pelos primeiros 
𝑘
k componentes seja muito próxima do total. Esse critério pode ser implementado da seguinte forma:

Variância Acumulada: Calcule a variância acumulada dos valores singulares e identifique o menor 
𝑘
k para o qual essa variância acumulada excede uma certa porcentagem, como 99% ou 100%.
Critério do Limite de Variância: Defina uma variável variance_threshold com o valor máximo permitido (como 1.0 para 100% da variância) e selecione 
𝑘
k quando a variância acumulada excede essa porcentagem do total."""
    def _matrix_Q_SVD(self, variance_threshold=0.7):
        # Realiza a decomposição SVD
        self.U, S, self.Vt = np.linalg.svd(self.Q, full_matrices=True, compute_uv=True)
        
        if not self.K_component:
            # Calcula a soma total dos valores singulares
            total_variance = np.sum(S)    
            # Calcula a soma acumulada dos valores singulares
            cumulative_variance = np.cumsum(S)
            
            # Encontra o k que explica a porcentagem de variação desejada
            k = np.argmax(cumulative_variance >= variance_threshold * total_variance) + 1
            self.S = np.diag(S)
            # Chama a função para selecionar os k primeiros componentes principais
            self._SVD_k_componente_pinciapl(k)
            self.K_component = True
        else:
            self.S = np.diag(S)
        
        

    def _SVD_k_componente_pinciapl(self, k):
        # Seleciona os primeiros k componentes principais
        self.U = self.U[:, :k]  # Mantém as primeiras k colunas de U
        self.S = self.S[:k, :k]  # Mantém a diagonal dos primeiros k valores singulares (matriz S_k)
        self.Vt = self.Vt[:k, :]  # Mantém as primeiras k linhas de Vt
        
    def _initialize_model(self, x, y):
        kss = self._calculate_kss(x)

        # Initialize model parameters (Equations 9A, 9B, and 9C)
        self.mu = self._calculate_initial_mu(y, kss)
        self.Sigma = self._calculate_initial_sigma(kss)
        self.Q = self._calculate_initial_Q(kss)
        if(self.using_SVD):
            self._matrix_Q_SVD() # ----> inserir a svd aqui
        # Initialize basic variables
        self._initialize_basis_and_samples(x)

        # Maximum likelihood estimates
        self._initialize_mle(y, kss)

        # Mark the model as initialized
        self._is_init = True

    def _calculate_kss(self, x):
        """Calculates kss adding jitter for numerical stability."""
        return self._kernel(x) + self.nu

    def _calculate_initial_mu(self, y, kss):
        """Calculates the initial mean (EQUATION 9A)."""
        return (y * kss) / (kss + self.c)

    def _calculate_initial_sigma(self, kss):
        """Calculates the initial covariance (EQUATION 9B)."""
        return kss - (kss ** 2) / (kss + self.c)

    def _calculate_initial_Q(self, kss):
        """Calculates the initial Q matrix (EQUATION 9C)."""
        return 1 / kss

    def _initialize_basis_and_samples(self, x):
        """Initializes the basis and sample points."""
        self.basis = 0
        self.Xb = x
        self.m = 1

    def _initialize_mle(self, y, kss):
        """Initializes maximum likelihood estimates."""
        self.nums02ML = y ** 2 / (kss + self.c)
        self.dens02ML = 1
        self.s02 = self.nums02ML / self.dens02ML

        # Forgetting mechanism
    def _apply_forgetting(self):
        if self._lambda < 1:
            if self.forgetmode == "B2P":  # Back-to-prior
                Kt = self._kernel(self.Xb)
                self.Sigma = self._lambda * self.Sigma + (1 - self._lambda) * Kt
                self.mu = np.sqrt(self._lambda) * self.mu

    # Prediction step
    def _predict_new_sample(self, x):
        kbs = self._kernel(self.Xb, np.atleast_2d(x))
        kss = self._kernel(x) + self.nu

        #q = self.Q @ kbs
        if self.using_SVD:
            q = self.U@self.S@self.Vt @kbs # MODIFIQUEI AQUI
        else:
            q = self.Q @ kbs

        ymean = q.T @ self.mu
        gamma2 = np.maximum(kss - kbs.T @ q, 0)  # Ensure non-negative variance
        h = self.Sigma @ q
        sf2 = np.maximum(gamma2 + q.T @ h, 0)  # Ensure non-negative sf2
        sy2 = self.c + sf2
        return q, ymean, gamma2, h, sf2, sy2

    # Update model with new sample
    def _update_with_new_sample(self, x, y, t, q, ymean, gamma2, h, sf2, sy2):
        p_q = np.block([[q], [-1]])
        if self.using_SVD:
            self.Q = np.block([[self.U @ self.S @self.Vt, np.zeros((self.m, 1))], [np.zeros((1, self.m)), 0]]) + (1 / gamma2) * (p_q @ p_q.T)
            self._matrix_Q_SVD() # -----> SVD aqui

        else:
            if self.Q.size == 0: #Acrescentei isso
                # Inicialize self.Q como uma matriz compatível
                self.Q = np.zeros((self.m, self.m)) #isso tbm
            self.Q = np.block([[self.Q, np.zeros((self.m, 1))], [np.zeros((1, self.m)), 0]]) + (1 / gamma2) * (p_q @ p_q.T)

        p_h = np.block([[h], [sf2]])
        self.mu = np.block([[self.mu], [ymean]]) + ((y - ymean) / sy2) * p_h
        self.Sigma = np.block([[self.Sigma, h], [h.T, sf2]]) - (1 / sy2) * (p_h @ p_h.T)
        
        self.basis = np.block([[self.basis], [t]])
        self.Xb = np.block([[self.Xb], [x]])
        self.m += 1
    


    # Maximum likelihood estimate for s02
    def _estimate_s02_ml(self, y, ymean, sy2):
        self.nums02ML += self._lambda * (y - ymean) ** 2 / sy2
        self.dens02ML += self._lambda
        self.s02 = self.nums02ML / self.dens02ML

    # Pruning mechanism
    '''def _prune_basis(self, gamma2, Q_old):
        if self.m > self.M or gamma2 < self.nu:
            if gamma2 < self.nu:
                if gamma2 < self.nu / 10:
                    warnings.warn("Numerical roundoff error is too high. Try increasing jitter noise.")
                criterium = np.block([np.ones((self.m - 1)), 0])
            else:  # MSE pruning
                #errors = (self.Q @ self.mu).reshape(-1) / np.diag(self.Q)
                if self.Q.shape[0] != self.mu.shape[0]: # acrescentei esse if aqui
                    self.mu = np.resize(self.mu, (self.Q.shape[0], 1))  # Redimensione conforme necessário
                # MODIFIQUEI AQUI TBM
                if self.using_SVD:
                    errors = (self.U@self.S@self.Vt@self.mu).reshape(-1) / np.diag(self.U@self.S@self.Vt@self.mu)
                else:
                    errors = (self.Q @ self.mu).reshape(-1) / np.diag(self.Q)
                criterium = np.abs(errors)
            
            r = np.argmin(criterium)
            smaller = criterium > criterium[r]

            if r == self.m:  # Revert if removing the element just added
                self.Q = Q_old
                if self.using_SVD:
                    self._matrix_Q_SVD() # ----> SVD aqui
                 #self.U @ self.S @self.Vt
            else:
                
                if self.using_SVD:
                    Qs = (self.U @ self.S @self.Vt)[smaller, r]
                    qs = (self.U @ self.S @self.Vt)[r, r]
                    self.Q = (self.U @ self.S @self.Vt)[smaller][:, smaller] - (Qs.reshape(-1, 1) * Qs.reshape(1, -1)) / qs
                    self._matrix_Q_SVD() # -----> SVD aqui
                else:
                    Qs = self.Q[smaller, r]
                    qs = self.Q[r, r]
                    self.Q = self.Q[smaller][:, smaller] - (Qs.reshape(-1, 1) * Qs.reshape(1, -1)) / qs

            # Update after pruning
            self.mu = self.mu[smaller]
            self.Sigma = self.Sigma[smaller][:, smaller]
            self.basis = self.basis[smaller]
            self.Xb = self.Xb[smaller, :]
            self.m -= 1'''
    def _prune_basis(self, gamma2, Q_old):
        if self.m > self.M or gamma2 < self.nu:
            if gamma2 < self.nu:
                if gamma2 < self.nu / 10:
                    warnings.warn("Numerical roundoff error is too high. Try increasing jitter noise.")
                criterium = np.block([np.ones((self.m - 1)), 0])
            else:  # MSE pruning
                # Verifique as dimensões antes de calcular os erros
                if self.Q.shape[0] != self.mu.shape[0]:
                    # Garantir que self.mu tenha o mesmo número de linhas de self.Q
                    self.mu = np.resize(self.mu, (self.Q.shape[0], 1))  # Redimensione conforme necessário

                if self.using_SVD:
                    errors = (self.U @ self.S @ self.Vt @ self.mu).reshape(-1) / np.diag(self.U @ self.S @ self.Vt @ self.mu)
                else:
                    errors = (self.Q @ self.mu).reshape(-1) / np.diag(self.Q)

                criterium = np.abs(errors)

            r = np.argmin(criterium)
            smaller = criterium > criterium[r]

            if r == self.m:  # Revert if removing the element just added
                self.Q = Q_old
                if self.using_SVD:
                    self._matrix_Q_SVD()  # ----> SVD aqui
            else:
                if self.using_SVD:
                    Qs = (self.U @ self.S @ self.Vt)[smaller, r]
                    qs = (self.U @ self.S @ self.Vt)[r, r]
                    self.Q = (self.U @ self.S @ self.Vt)[smaller][:, smaller] - (Qs.reshape(-1, 1) * Qs.reshape(1, -1)) / qs
                    self._matrix_Q_SVD()  # ----> SVD aqui
                else:
                    Qs = self.Q[smaller, r]
                    qs = self.Q[r, r]
                    self.Q = self.Q[smaller][:, smaller] - (Qs.reshape(-1, 1) * Qs.reshape(1, -1)) / qs

            # Update after pruning, garantir o índice booleano correto
            self.mu = self.mu[smaller]
            
            # Aqui você deve garantir que a indexação booleana está sendo aplicada corretamente
            self.Sigma = self.Sigma[smaller, :][:, smaller]  # Ajuste de indexação para Sigma 2D
            self.basis = self.basis[smaller]
            self.Xb = self.Xb[smaller, :]  # Se for uma matriz 2D, use Xb[smaller, :]
            
            self.m -= 1

    # Main function call
    def process_sample(self, x, y, t):
        self._apply_forgetting()
        q, ymean, gamma2, h, sf2, sy2 = self._predict_new_sample(x)
        Q_old = self.Q.copy()
        
        self._update_with_new_sample(x, y, t, q, ymean, gamma2, h, sf2, sy2)
        self._estimate_s02_ml(y, ymean, sy2)
        #self._prune_basis(gamma2, Q_old)
        if self.using_SVD:
            if self.mu.shape[0] != self.Q.shape[0]: #Acrescentei isso
                self.mu = np.resize(self.mu, (self.Q.shape[0], 1)) #isso tbm
            self._prune_basis(gamma2, self.U@self.S@self.Vt@self.mu) # MODIFIQUEI AQUI
        else:
            self._prune_basis(gamma2, Q_old)

    def learn_one(self, x: np.array, y:np.array, t: int):
        """
        Function to learn a given and an instant of time t.

        Args:
            x (np.array): Features
            y (np.array): Target
            t (int): index of time
        """

        if not self._is_init:
            self._initialize_model(x, y)
        else:  # Update model
            self.process_sample(x,y,t)
    
    def partial_fit(self, X: np.ndarray, Y: np.ndarray, T: np.ndarray):
        """
        Function to learn the training dataset.

        Args:
            x (np.array): Features
            y (np.array): Target
            t (int): index of time
        """
        for x, y, t in zip(X, Y, T):
            self.learn_one(x, y, t) 
     
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Predicts mean and variance for potentially unseen data X.

        Args:
            X (np.ndarray): Array of data points with shape (n_data_points, n_features)

        Returns:
            Tuple[np.ndarray, np.ndarray]: Predicted mean and variance as (mean_est, var_est)
        """
        # Calculate the kernel values between the basis points and new data points
        kbs = self._kernel(self.Xb, np.atleast_2d(X))
        
        # Predict the mean
        mean_est = self._predict_mean(kbs)

        # Predict the variance
        sf2 = self._calculate_sf2(kbs)
        var_est = self._predict_variance(sf2)

        return mean_est, var_est

    def _predict_mean(self, kbs: np.ndarray) -> np.ndarray:
        """Calculates the predicted mean for the input data.
        
        Args:
            kbs (np.ndarray): Kernel values between basis points and input data.
        
        Returns:
            np.ndarray: Predicted mean.
        """
        #return kbs.T @ self.Q @ self.mu
        if self.using_SVD:
            return kbs.T @ self.U @ self.S @ self.Vt @ self.mu  # MODIFIQUEI AQUI
        else:
            return kbs.T @ self.Q @ self.mu

    def _calculate_sf2(self, kbs: np.ndarray) -> np.ndarray:
        """Calculates the squared scaling factor sf2 for variance prediction.
        
        Args:
            kbs (np.ndarray): Kernel values between basis points and input data.
        
        Returns:
            np.ndarray: Squared scaling factor for variance prediction.
        """
        if self.using_SVD:
            result = (self.U @ self.S @ self.Vt @ self.Sigma @ self.U @ self.S @ self.Vt - self.U @ self.S @ self.Vt)
        else:
            result = (self.Q @ self.Sigma @ self.Q - self.Q)
        sf2 = (
            1
            + self.nu
            + np.sum(
                kbs * (result @ kbs), axis=0
            ).reshape(-1, 1)
        )
        sf2[sf2 < 0] = 0  # Ensure non-negative variance

        return sf2

    def _predict_variance(self, sf2: np.ndarray) -> np.ndarray:
        """Calculates the predicted variance for the input data.
        
        Args:
            sf2 (np.ndarray): Squared scaling factor for variance.
        
        Returns:
            np.ndarray: Predicted variance.
        """
        return self.s02 * (self.c + sf2)