import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt


def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=1,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup_resolution = 10
        self.lam = 0.95
        self.lam_r = 0.8
        self.eps_nan = 1e-4
        self.zeta = 0.4
        
        self.init_alphas = np.array(alphas_init, dtype=np.float64)
        self.init_betas = np.array(betas_init, dtype=np.float64)
        self.init_weight = np.array(weights_init, dtype=np.float64)
        
    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])
    
    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x, y=None):
        if y is None:
            r = np.array([self.weighted_likelihood(x, i) for i in range(2)])
        else:
            r = self.lam_r * np.array([self.weighted_likelihood(x, i) for i in range(2)])
            r = (1 - self.lam_r) * np.eye(2)[y].T

        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= (r.sum(axis=0) + 1e-12)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x, y=None):
        x = np.copy(x)

        # EM on beta distributions unable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        if y is not None: return self.fit_supervised(x, y)
        else: return self.fit_unsupervised(x)
    
    def fit_supervised(self, x, y):
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x, y)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            
        return self
        
    def fit_unsupervised(self, x):
        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)

            # M-step
            for i in range(2):
                new_alpha, new_beta = fit_beta_weighted(x, r[i])
                self.alphas[i] = self.lam * self.alphas[i] + (1 - self.lam) * new_alpha
                self.betas[i] = self.lam * self.betas[i] + (1 - self.lam) * new_beta

            new_weight = r.sum(axis=1)
            new_weight /= (new_weight.sum() + 1e-12)
            self.weight = self.lam * self.weight + (1-self.lam) * new_weight.reshape(-1)
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        
    def predict_proba(self, xmin=0.0, xmax=1.0):
        x = np.linspace(xmin, xmax, self.lookup_resolution)
        mean = self.alphas / (self.alphas + self.betas)
        i = mean.argmax()
        pred = self.likelihood(x, i) / (self.likelihood(x, i) + self.likelihood(x, 1-i)) > self.zeta
        idx = min(max(0, self.lookup_resolution - sum(pred)), self.lookup_resolution-1)
        return x[idx]
    
    def reinit(self):
        self.alphas = np.copy(self.init_alphas)
        self.betas = np.copy(self.init_betas)
        self.weight = np.copy(self.init_weight)

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
