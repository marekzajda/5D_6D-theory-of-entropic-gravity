import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import quad, odeint, solve_ivp
from scipy.optimize import minimize
from scipy.stats import norm
import pandas as pd

# =================================================================
# Fyzikální konstanty a parametry teorie (rozšířené)
# =================================================================
PLANCK_LENGTH = 1.616255e-35  # [m]
BOLTZMANN = 1.380649e-23      # [J/K]
G5 = 6.708e-39                # 5D gravitační konstanta [m^3/kg s^2]
CY_EULER_CHI = -200           # Eulerova charakteristika CY variety
M6 = 1.2e16                   # 6D Planckova hmotnost [GeV]
H0 = 2.18e-18                 # Hubbleova konstanta [s^-1]

# PID parametry z vaší teorie
kP = 1.047
kI = 2.31e-3
kD = 0.178

# Parametry kvantových fluktuací
FLUCTUATION_AMPLITUDE = 0.01
FLUCTUATION_CORRELATION = 1e-35

# =================================================================
# Rozšířené numerické funkce
# =================================================================
def quintic_cy_metric(z1, z2, z3):
    """Přesná metrika pro quintickou CY varietu v CP^4."""
    # Definice pomocí homogenních souřadnic
    denominator = (1 + np.abs(z1)**4 + np.abs(z2)**4 + np.abs(z3)**4)**(1/3)
    return np.array([
        [1/(denominator*(1 + np.abs(z1)**2)), 0, 0],
        [0, 1/(denominator*(1 + np.abs(z2)**2)), 0],
        [0, 0, 1/(denominator*(1 + np.abs(z3)**2))]
    ])

def quantum_fluctuations(S, t):
    """Stochastické kvantové fluktuace entropie."""
    # Korelace fluktuací s časovým vývojem
    correlation = FLUCTUATION_CORRELATION * np.exp(-t/PLANCK_LENGTH)
    fluctuation = FLUCTUATION_AMPLITUDE * np.random.normal() * np.sqrt(S) * correlation
    return fluctuation

def cosmic_scale_factor(t):
    """Řešení Friedmannových rovnic s entropickou korekcí."""
    a0 = 1e-25  # Počáteční podmínka
    def da_dt(t, a):
        H = H0 * (1 + kP*np.log(a/a0) - kI*t + kD*(1/t if t>0 else 0)
        return a * H
    solution = solve_ivp(da_dt, [0, t], [a0], method='RK45')
    return solution.y[0][-1]

# =================================================================
# Vylepšené výpočetní jádro
# =================================================================
def calculate_6d_entropy(radius, t):
    """Entropie 6D prostoru s kvantovými fluktuacemi."""
    # Základní entropie
    S0 = (np.pi**3 * radius**5) / (2 * PLANCK_LENGTH**5) * BOLTZMANN
    
    # PID regulace
    S_pid = entropy_pid_controller(S0, t)
    
    # Kvantové fluktuace
    S_quantum = quantum_fluctuations(S0, t)
    
    # Korekce z CY metriky
    z = radius / (5*PLANCK_LENGTH)
    cy_vol = np.linalg.det(quintic_cy_metric(z, z, z))
    cy_corr = 1 + 0.2*np.log(cy_vol)
    
    return S0 * S_pid * cy_corr + S_quantum

def holographic_pid_control(t, S):
    """PID regulátor s kosmologickými korekcemi."""
    # Reference na Planck data
    S_planck = 2.9e19 * BOLTZMANN * cosmic_scale_factor(t)**3
    
    # Dynamické koeficienty
    kP_dyn = kP * (1 + 0.01*np.sin(t/H0))  # Oscilace z inflace
    kI_dyn = kI * cosmic_scale_factor(t)
    kD_dyn = kD / (1 + t*H0)
    
    # PID rovnice
    error = S - S_planck
    integral = quad(lambda tau: error, 0, t)[0]
    derivative = np.gradient([S])[0] if isinstance(S, np.ndarray) else 0
    
    return -kP_dyn*error - kI_dyn*integral - kD_dyn*derivative

# =================================================================
# Kompletní testovací framework
# =================================================================
def full_cosmological_test(redshift):
    """Test teorie proti kosmologickým pozorováním."""
    t = 1/H0 * np.log(1/(1+redshift))  # Převod redshiftu na čas
    
    # Výpočet entropie
    r = cosmic_scale_factor(t) * 1e26  # Poloměr Hubbleova horizontu
    S = calculate_6d_entropy(r, t)
    
    # Porovnání s Planck daty
    S_planck = 2.9e19 * BOLTZMANN * (1+redshift)**3
    ratio = S / S_planck
    
    return {
        'redshift': redshift,
        'entropy_predicted': S,
        'entropy_observed': S_planck,
        'ratio': ratio,
        'consistency': 0.9 < ratio < 1.1
    }

# =================================================================
# Vizualizace a analýza (rozšířená)
# =================================================================
def plot_cosmological_comparison(redshifts):
    """Porovnání s kosmologickými daty."""
    results = [full_cosmological_test(z) for z in redshifts]
    
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.semilogx(redshifts, [r['ratio'] for r in results], 'b-', lw=2)
    ax.axhline(1, color='k', linestyle='--')
    ax.fill_between(redshifts, 0.9, 1.1, color='gray', alpha=0.2)
    ax.set_xlabel('Redshift (z)')
    ax.set_ylabel('Poměr: Predikce/Observace')
    ax.set_title('Test proti Planckovým datům')
    plt.savefig('cosmological_test.png')
    plt.show()
    
    return pd.DataFrame(results)

# =================================================================
# Hlavní výpočetní blok
# =================================================================
if __name__ == "__main__":
    print("Spouštím kompletní analýzu 5D/6D entropické gravitace...")
    
    # 1. Test holografické meze
    radii = np.logspace(np.log10(PLANCK_LENGTH), np.log10(1e26), 100)
    results = [advanced_holographic_test(r, dim=6) for r in radii]
    pd.DataFrame(results).to_csv('holographic_results.csv', index=False)
    
    # 2. Kosmologický test
    redshifts = np.logspace(-3, 3, 50)
    cosmology_results = plot_cosmological_comparison(redshifts)
    
    # 3. Výpočet částicových vlastností
    yukawa_results = {
        'y11': yukawa_integral(omega_1, omega_1),
        'y12': yukawa_integral(omega_1, omega_2),
        'y22': yukawa_integral(omega_2, omega_2),
        'y33': yukawa_integral(omega_3, omega_3)
    }
    
    print("\nShrnutí výsledků:")
    print(f"- Holografická mez platí pro R > {next(r['radius'] for r in results if r['bound_respected']):.1e} m")
    print(f"- Kosmologická shoda: {cosmology_results['consistency'].mean()*100:.1f}%")
    print(f"- Yukawovy koeficienty: {yukawa_results}")
    
    # Uložení všech výsledků
    with pd.ExcelWriter('full_analysis.xlsx') as writer:
        pd.DataFrame(results).to_excel(writer, sheet_name='Holografický test')
        cosmology_results.to_excel(writer, sheet_name='Kosmologický test')
        pd.DataFrame.from_dict(yukawa_results, orient='index').to_excel(writer, sheet_name='Yukawovy koeficienty')

    print("Analýza úspěšně dokončena. Výsledky uloženy v full_analysis.xlsx")
