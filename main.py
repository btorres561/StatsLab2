import numpy as np
import matplotlib.pyplot as plt
from scipy import stats


def problem1(x_unif, x_exp, x_norm):
    #Plot histogram and true PDF overlay for each distribution
    t_u = np.linspace(0, 4, 300)
    plt.hist(x_unif, density=True, bins=50, alpha=0.6)
    plt.plot(t_u, stats.uniform.pdf(t_u, loc=0, scale=4), 'r-')
    plt.title('Uniform(0,4)')
    plt.show()

    t_e = np.linspace(0, 15, 300)
    plt.hist(x_exp, density=True, bins=50, alpha=0.6)
    plt.plot(t_e, stats.expon.pdf(t_e, scale=2), 'r-')
    plt.title('Exponential(λ=0.5)')
    plt.show()

    t_n = np.linspace(-2, 6, 300)
    plt.hist(x_norm, density=True, bins=50, alpha=0.6)
    plt.plot(t_n, stats.norm.pdf(t_n, loc=2, scale=1), 'r-')
    plt.title('Normal(2,1)')
    plt.show()

    #Overlay exponential PDFs for 3 different rates on the same axes
    for lam in [0.25, 0.5, 1]:
        plt.plot(t_e, stats.expon.pdf(t_e, scale=1/lam), label=f'λ={lam}')
    plt.legend()
    plt.show()

    #Estimate P(XE > 3) from sample and compare to exact value e^(-1.5)
    print(f"\nP(XE>3) simulated: {np.mean(x_exp > 3):.4f}, exact: {np.exp(-1.5):.4f}")


def problem2():
    #Compute Φ(1.5) and Φ(-1.5)
    print(f"\nΦ(1.5)={stats.norm.cdf(1.5):.4f}, Φ(-1.5)={stats.norm.cdf(-1.5):.4f}")

    #P(85 < X < 130) = Φ(2) - Φ(-1) after standardizing
    print(f"\nP(85<X<130) = {stats.norm.cdf(2) - stats.norm.cdf(-1):.4f}")

    #90th percentile: find z, then convert to X scale
    print(f"\n90th percentile: {100 + 15 * stats.norm.ppf(0.90):.4f}")

    #68-95-99.7 rule: P(|Z| < k) for k = 1, 2, 3
    for k in [1, 2, 3]:
        print(f"P(|Z|<{k}) = {stats.norm.cdf(k) - stats.norm.cdf(-k):.4f}")

    #Simulate 100,000 values and estimate P(85 < X < 130)
    np.random.seed(1)
    x_sim = np.random.normal(100, 15, 100_000)
    print(f"\nP(85<X<130) simulated: {np.mean((x_sim > 85) & (x_sim < 130)):.4f}")


def problem3(x_unif, x_exp, x_norm):
    #Print theoretical vs sample mean and variance for each distribution
    for name, x, theo_mean, theo_var in [
        ('XU', x_unif, 2.0,   (4**2)/12),
        ('XE', x_exp,  1/0.5, 1/0.5**2),
        ('XN', x_norm, 2.0,   1.0),
    ]:
        print(f"{name}: E[X]={theo_mean}, Sample Mean={np.mean(x):.4f}, Var(X)={theo_var}, Sample Var={np.var(x, ddof=1):.4f}")

    #Verify Var(X) = E[X^2] - E[X]^2 using x_exp
    e_x2 = np.mean(x_exp**2)
    print(f"\nE[XE^2]={e_x2:.4f}, E[X^2]-E[X]^2={e_x2 - (1/0.5)**2:.4f}, Var(XE)={1/0.5**2:.4f}")


def main():
    #Draw 10,000 samples from each distribution
    np.random.seed(1)
    n = 10000
    x_unif = np.random.uniform(0, 4, n)
    x_exp = np.random.exponential(scale=1/0.5, size=n)
    x_norm = np.random.normal(2, 1, n)

    problem1(x_unif, x_exp, x_norm)
    problem2()
    problem3(x_unif, x_exp, x_norm)


main()
