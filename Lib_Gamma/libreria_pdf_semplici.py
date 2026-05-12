'''LIBRERIA CON MODELLI PDF SEMPLICI, FIT E ANALISI PER SPETTROSCOPIA GAMMA''' 

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.special import erfc
from scipy.signal import find_peaks
from scipy.stats import chi2
from scipy.stats import f 
from iminuit import Minuit
from iminuit.cost import LeastSquares
import Lib_Gamma.libreria_funzioni_utili as lfu  # libreria creata, nella cartella "Librerie"

# =========================================================
# 1. DEFINIZIONE DEI MODELLI FISICI "semplici"
# =========================================================

def gauss_costante_model(x, a, mu, sigma, c):
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c

def gauss_retta_model(x, a, mu, sigma, c, b):
    # Fondo lineare: c + b*(x - mu). Centrare in mu riduce la correlazione tra parametri!
    return a * np.exp(-(x - mu)**2 / (2 * sigma**2)) + c + b * (x - mu)

def gauss_step_cost_model(x, a, mu, sigma, c, S):
    # Gaussiana + Costante + Gradino (Step). erfc vale 2 a sx del picco, 1 al centro, 0 a dx.
    gauss = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    step = S * erfc((x - mu) / (np.sqrt(2) * sigma))
    return gauss + c + step

def gauss_step_retta_model(x, a, mu, sigma, c, b, S):
    # Gaussiana + Retta + Gradino Compton
    gauss = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    fondo_lin = c + b * (x - mu)
    step = S * erfc((x - mu) / (np.sqrt(2) * sigma))
    return gauss + fondo_lin + step

# --- MODELLI CON CODE ESPONENZIALI (Spegnimento ai bordi tramite erfc) ---

def gauss_exp_sx_cost_model(x, a, mu, sigma, c, A_sx, k_sx):
    gauss = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    # np.clip evita che l'esponenziale esploda a infinito causando NaN quando erfc vale 0
    arg_exp = np.clip(k_sx * (x - mu), -250, 250)
    tail_sx = A_sx * np.exp(arg_exp) * (erfc((x - mu) / (np.sqrt(2) * sigma)) / 2)
    return gauss + c + tail_sx

def gauss_exp_sx_retta_model(x, a, mu, sigma, c, b, A_sx, k_sx):
    gauss = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    fondo_lin = c + b * (x - mu)
    arg_exp = np.clip(k_sx * (x - mu), -250, 250)
    tail_sx = A_sx * np.exp(arg_exp) * (erfc((x - mu) / (np.sqrt(2) * sigma)) / 2)
    return gauss + fondo_lin + tail_sx

def gauss_exp_dx_cost_model(x, a, mu, sigma, c, A_dx, k_dx):
    gauss = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    arg_exp = np.clip(-k_dx * (x - mu), -250, 250)
    tail_dx = A_dx * np.exp(arg_exp) * (erfc(-(x - mu) / (np.sqrt(2) * sigma)) / 2)
    return gauss + c + tail_dx

def gauss_exp_dx_retta_model(x, a, mu, sigma, c, b, A_dx, k_dx):
    gauss = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    fondo_lin = c + b * (x - mu)
    arg_exp = np.clip(-k_dx * (x - mu), -250, 250)
    tail_dx = A_dx * np.exp(arg_exp) * (erfc(-(x - mu) / (np.sqrt(2) * sigma)) / 2)
    return gauss + fondo_lin + tail_dx


# =========================================================
# FUNZIONE DI FIT GENERICA
# =========================================================

def fit_generic_model(roi_x, roi_y, err_y, model, guess_params, limits=None):
    """
    Funzione universale per fittare qualsiasi modello definito sopra usando iminuit.
    """
    cost = LeastSquares(roi_x, roi_y, err_y, model)
    # Minuit accetta i guess passati come keyword arguments spacchettando il dizionario
    m = Minuit(cost, **guess_params)
    
    # Applichiamo i limiti se forniti
    if limits:
        for param, lim in limits.items():
            m.limits[param] = lim
            
    m.migrad()
    m.hesse()
    return m


# ================================================================================================
# Calcola la FWHM effettiva (numerica) del fotopicco e ne stima l'incertezza (tecnica Monte Carlo)
# ================================================================================================

def calcola_fwhm_effettiva(m_fit, roi_x, n_samples=1000):
    """
    Calcola la FWHM effettiva (numerica) del fotopicco e ne stima l'incertezza (-> "Parametric Bootstrap")
    
    PERCHÉ UN CALCOLO NUMERICO?
    La formula analitica FWHM = 2.355 * sigma vale SOLO per una gaussiana perfetta.
    Se il fit vincente include code esponenziali, il picco è asimmetrico e si allarga.
    Questa funzione ricostruisce il segnale e ne misura la larghezza a metà altezza
    direttamente sull'asse X.
    
    PERCHÉ IL MONTE CARLO PER L'ERRORE?
    La FWHM numerica dipende da parametri fortemente correlati tra loro (sigma, 
    ampiezze e costanti di decadimento delle code). Non potendo usare la normale 
    propagazione degli errori, usiamo la matrice di covarianza per generare 
    'n_samples' set di parametri statisticamente equivalenti, misuriamo la FWHM 
    per ciascuno, e prendiamo la deviazione standard della distribuzione risultante.
    """
    
    # Creiamo un asse X ad altissima risoluzione (10000 punti) per garantire 
    # che la ricerca numerica della "metà altezza" sia estremamente precisa.
    x_dense = np.linspace(roi_x[0], roi_x[-1], 10000)
    
    # Estraiamo i parametri ottimali (best-fit) in formato dizionario
    v_dict = m_fit.values.to_dict()
    
    # =========================================================================
    # FUNZIONE INTERNA: Calcola la FWHM numerica dato un set di parametri
    # =========================================================================
    def get_fwhm_from_params(v):
        # 1. Ricostruzione del Segnale Fisico:
        # Partiamo dalla componente gaussiana pura (il core del fotopicco)
        segnale = v['a'] * np.exp(-(x_dense - v['mu'])**2 / (2 * v['sigma']**2))
        
        # 2. Aggiunta delle code al segnale:
        # Il tailing (incompleta raccolta di carica) fa parte dei conteggi del picco.
        # La funzione erfc agisce da "interruttore": accende l'esponenziale solo da un lato.
        if 'A_sx' in v and 'k_sx' in v:
            arg_exp = np.clip(v['k_sx'] * (x_dense - v['mu']), -250, 250)
            segnale += v['A_sx'] * np.exp(arg_exp) * (erfc((x_dense - v['mu']) / (np.sqrt(2) * v['sigma'])) / 2)

        if 'A_dx' in v and 'k_dx' in v:
            arg_exp = np.clip(-v['k_dx'] * (x_dense - v['mu']), -250, 250)
            segnale += v['A_dx'] * np.exp(arg_exp) * (erfc(-(x_dense - v['mu']) / (np.sqrt(2) * v['sigma'])) / 2)
                        
        # 3. Misura della FWHM:
        max_y = np.max(segnale)            # Trova l'altezza massima del picco
        half_max = max_y / 2.0             # Calcola la metà altezza
        
        # Trova tutti gli indici in cui la curva supera la metà altezza
        above_hm = np.where(segnale >= half_max)[0]
        
        # Sicurezza: se la curva degenera (es. parametri strani estratti dal MC), ritorna 0
        if len(above_hm) < 2:
            return 0 
            
        # La FWHM è la distanza sull'asse X tra l'ultimo e il primo punto sopra la metà altezza
        return x_dense[above_hm[-1]] - x_dense[above_hm[0]]
    # =========================================================================

    # --- A. CALCOLO FWHM NOMINALE ---
    # Calcoliamo la larghezza usando esattamente i parametri di best-fit trovati da Minuit
    fwhm_effettiva = get_fwhm_from_params(v_dict)
    
    # --- B. CALCOLO INCERTEZZA FWHM (TOY MONTE CARLO) -> "Parametric Bootstrap"---
    fwhm_err = 0.0
    
    # Eseguiamo il Monte Carlo solo se il fit è valido e abbiamo la matrice di covarianza
    if m_fit.valid and m_fit.covariance is not None:
        param_names = list(m_fit.parameters)
        best_vals = [m_fit.values[p] for p in param_names]
        cov_matrix = m_fit.covariance
        
        try:
            # np.random.multivariate_normal estrae vettori di parametri casuali,
            # rispettando scrupolosamente le incertezze e le correlazioni tra i parametri.
            random_params = np.random.multivariate_normal(best_vals, cov_matrix, n_samples)
            fwhms_mc = []
            
            # Calcoliamo la FWHM per ogni set di parametri "perturbato"
            for pars in random_params:
                v_random = dict(zip(param_names, pars))
                fwhms_mc.append(get_fwhm_from_params(v_random))
                
            # L'incertezza sulla FWHM è la deviazione standard campionaria delle FWHM simulate
            fwhm_err = np.std(fwhms_mc, ddof=1)
        except Exception as e:
            print(f"  [Attenzione] Impossibile eseguire il Monte Carlo per l'errore FWHM: {e}")

    return fwhm_effettiva, fwhm_err


# =====================================================================
# STUDIO STABILITÀ WINDOW
# ===================================================================== 

def studio_stabilita_window(data, channels, picco_ch, nome_picco, win_min=25, win_max=80, step=2):
    """
    Studia come varia la FWHM e la sua incertezza al variare della Region Of Interest (window).
    Esegue il fit in modalità "silenziosa" per non riempire lo schermo.
    """
    windows_array = np.arange(win_min, win_max + 1, step)
    fwhm_list = []
    err_list = []
    win_list = []
    
    print(f"Avvio studio stabilità per il picco: {nome_picco} (Canale ~{picco_ch})")
    print(f"Scansione window da {win_min} a {win_max}... attendere.")
    
    for w in windows_array:
        # 1. Estrazione ROI
        roi_x = channels[picco_ch - w : picco_ch + w]
        roi_y = data[picco_ch - w : picco_ch + w]
        err_y = np.sqrt(roi_y)
        err_y[err_y == 0] = 1.0
        
        # 2. Guesses di base
        fondo_sx = np.mean(roi_y[:5])
        fondo_dx = np.mean(roi_y[-5:])
        fondo_medio = (fondo_sx + fondo_dx) / 2
        amp = max(10, data[picco_ch] - fondo_medio)
        pend = (fondo_dx - fondo_sx) / (2 * w)
        
        # 3. FIT 1: Gaussiana + Retta (Modello Base solido)
        guess_M1 = {'a': amp, 'mu': picco_ch, 'sigma': 10, 'c': fondo_medio, 'b': pend}
        lim_M1 = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None)}
        m1 = fit_generic_model(roi_x, roi_y, err_y, gauss_retta_model, guess_M1, lim_M1)
        
        # 4. FIT 2: Gaussiana + Retta + Coda SX (Modello Complesso frequente per HPGe)
        guess_exp = {'a': amp, 'mu': picco_ch, 'sigma': 10, 'c': fondo_medio, 'b': pend, 'A_sx': amp*0.05, 'k_sx': 0.05}
        lim_exp = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None), 'A_sx': (0, None), 'k_sx': (0.001, None)}
        m_exp = fit_generic_model(roi_x, roi_y, err_y, gauss_exp_sx_retta_model, guess_exp, lim_exp)
        
        # 5. TEST-F Silenzioso
        chi2_1, ndof_1 = m1.fval, m1.ndof
        chi2_2, ndof_2 = m_exp.fval, m_exp.ndof
        delta_chi2 = chi2_1 - chi2_2
        delta_ndof = ndof_1 - ndof_2
        
        best_m = m1 # Default
        if delta_chi2 > 0 and delta_ndof > 0:
            f_val = (delta_chi2 / delta_ndof) / (chi2_2 / ndof_2)
            p_val = f.sf(f_val, delta_ndof, ndof_2)
            if p_val < 0.05:
                best_m = m_exp
                
        # 6. Estrazione FWHM (riduciamo i sample del MC a 200 per velocizzare il ciclo)
        fwhm, fwhm_err = calcola_fwhm_effettiva(best_m, roi_x, n_samples=200)
        
        # Salva solo se il fit non è crashato
        if fwhm > 0:
            win_list.append(w)
            fwhm_list.append(fwhm)
            err_list.append(fwhm_err)

    # ---------------------------------------------------------
    # Plot dei risultati
    # ---------------------------------------------------------
    plt.figure(figsize=(10, 6))
    plt.errorbar(win_list, fwhm_list, yerr=err_list, fmt='o-', color='teal', 
                 ecolor='darkslategray', capsize=3, markersize=5, linewidth=1.5)
    
    plt.title(f"Studio Stabilità Window - {nome_picco}", fontsize=14)
    plt.xlabel("Dimensione Window (canali sx/dx)", fontsize=12)
    plt.ylabel("FWHM Effettiva [canali]", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Evidenziamo un'ipotetica zona di plateau
    min_err_idx = np.argmin(err_list)
    miglior_win = win_list[min_err_idx]
    plt.axvline(miglior_win, color='red', linestyle='--', alpha=0.5, 
                label=f"Incertezza minima a Window={miglior_win}")
    
    plt.legend()
    plt.tight_layout()
    plt.show()
# --------------------------------------------------------------------
# ESEMPIO DI UTILIZZO (Sostituisci picco_ch con i canali veri!)
# --------------------------------------------------------------------
# studio_stabilita_window(data, channels, picco_ch=2975, nome_picco="Picco 511 keV", win_min=20, win_max=70)
# studio_stabilita_window(data, channels, picco_ch=7330, nome_picco="Picco 1274 keV", win_min=30, win_max=100)


#================================================================================
# TORNEO PER SELEZIONARE IL MIGLIOR MODELLO (TORNEO TEST-F), per Sodio22 con HPGe
#================================================================================
"""
Questa funzione seleziona il modello fisicamente e statisticamente più accurato 
per il fit dei fotopicchi HPGe, minimizzando il rischio di overfitting. 
Poiché il Test F di Fisher è rigorosamente valido solo per modelli "annidati", 
la selezione avviene attraverso un torneo a bivi logici:

1. Qualifica Base: Costante vs. Retta. Il Test-F stabilisce il fondo continuo locale.
2. Sfide Indipendenti: La Base vincente viene sfidata separatamente dall'aggiunta 
   di un Gradino Compton (funzione erfc) e di una Coda Esponenziale (tailing).
3. Spareggio Finale: Se entrambe le componenti extra migliorano significativamente 
   il fit base, i due modelli risultanti (non essendo annidati tra loro) vengono 
   confrontati e vince quello con il minor Chi-quadro ridotto (χ²/ndof).
"""

def best_easy_model_analisi_sodio22(data, channels, window1=55, window2=65, n_first_channels_to_ignore=200, prominence=200, distance=500):
    
    lfu.istogramma(data, channels)
    
    picchi_ch = lfu.auto_ricerca_n_picchi(data, 2, n_first_channels_to_ignore, prominence, distance)
    
    window_list = [window1, window2]
    best_models = []
    best_names = []
    roi_data = []

    nomi_picchi = ["Annichilazione 511 keV", "Decadimento 1274.5 keV"]
    
    for idx, picco_ch in enumerate(picchi_ch):
        print(f"\n{'='*70}\n ANALISI PICCO: {nomi_picchi[idx]} (Canale ~{picco_ch})\n{'='*70}")
        
        # 1. Estrazione ROI
        roi_x = channels[picco_ch - window_list[idx] : picco_ch + window_list[idx]]
        roi_y = data[picco_ch - window_list[idx] : picco_ch + window_list[idx]]
        err_y = np.sqrt(roi_y) # incertezza poissoniana
        err_y[err_y == 0] = 1.0
        
        # 2. Stime Iniziali 
        fondo_sx = np.mean(roi_y[:5])  
        fondo_dx = np.mean(roi_y[-5:]) 
        fondo_medio = (fondo_sx + fondo_dx) / 2
        ampiezza_gauss = max(10, data[picco_ch] - fondo_medio)
        pendenza_stimata = (fondo_dx - fondo_sx) / (2 * window_list[idx])
        gradino_stimato = max(0, (fondo_sx - fondo_dx) / 2)
        
        # --- M0: Gauss + Costante ---
        guess_M0 = {'a': ampiezza_gauss, 'mu': picco_ch, 'sigma': 10, 'c': fondo_medio}
        lim_M0 = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None)}
        m0 = fit_generic_model(roi_x, roi_y, err_y, gauss_costante_model, guess_M0, lim_M0)
        
        # --- M1: Gauss + Retta ---
        guess_M1 = {'a': ampiezza_gauss, 'mu': picco_ch, 'sigma': 10, 'c': fondo_medio, 'b': pendenza_stimata}
        lim_M1 = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None)}
        m1 = fit_generic_model(roi_x, roi_y, err_y, gauss_retta_model, guess_M1, lim_M1)
        
        # QUALIFICA BASE: Costante vs Retta
        print("\n[TEST 1: Base Lineare]")
        best_base = lfu.confronta_modelli_ftest(m0, m1, alpha=0.05, nome_picco=f"{nomi_picchi[idx]} (Cost vs Retta)")
        
        # Prepariamo Step, Exp SX e Exp DX a seconda del vincitore Base
        dict_modelli = {} # dizionario 
        
        if best_base == m0:
            dict_modelli[m0] = (gauss_costante_model, "Gaussiana + Costante")
            
            guess_step = {'a': ampiezza_gauss, 'mu': picco_ch, 'sigma': 10, 'c': fondo_dx, 'S': gradino_stimato}
            lim_step = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None), 'S': (0, None)}
            m_step = fit_generic_model(roi_x, roi_y, err_y, gauss_step_cost_model, guess_step, lim_step)
            dict_modelli[m_step] = (gauss_step_cost_model, "Gaussiana + Gradino + Costante")
            
            guess_exp_sx = {'a': ampiezza_gauss, 'mu': picco_ch, 'sigma': 10, 'c': fondo_medio, 'A_sx': ampiezza_gauss*0.05, 'k_sx': 0.05}
            lim_exp_sx = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None), 'A_sx': (0, None), 'k_sx': (0.001, None)}
            m_exp_sx = fit_generic_model(roi_x, roi_y, err_y, gauss_exp_sx_cost_model, guess_exp_sx, lim_exp_sx)
            dict_modelli[m_exp_sx] = (gauss_exp_sx_cost_model, "Gaussiana + Coda SX + Costante")

            guess_exp_dx = {'a': ampiezza_gauss, 'mu': picco_ch, 'sigma': 10, 'c': fondo_medio, 'A_dx': ampiezza_gauss*0.05, 'k_dx': 0.05}
            lim_exp_dx = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None), 'A_dx': (0, None), 'k_dx': (0.001, None)}
            m_exp_dx = fit_generic_model(roi_x, roi_y, err_y, gauss_exp_dx_cost_model, guess_exp_dx, lim_exp_dx)
            dict_modelli[m_exp_dx] = (gauss_exp_dx_cost_model, "Gaussiana + Coda DX + Costante")

        else:
            dict_modelli[m1] = (gauss_retta_model, "Gaussiana + Retta")
            
            guess_step = {'a': ampiezza_gauss, 'mu': picco_ch, 'sigma': 10, 'c': fondo_dx, 'b': pendenza_stimata, 'S': gradino_stimato}
            lim_step = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None), 'S': (0, None)}
            m_step = fit_generic_model(roi_x, roi_y, err_y, gauss_step_retta_model, guess_step, lim_step)
            dict_modelli[m_step] = (gauss_step_retta_model, "Gaussiana + Gradino + Retta")
            
            guess_exp_sx = {'a': ampiezza_gauss, 'mu': picco_ch, 'sigma': 10, 'c': fondo_medio, 'b': pendenza_stimata, 'A_sx': ampiezza_gauss*0.05, 'k_sx': 0.05}
            lim_exp_sx = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None), 'A_sx': (0, None), 'k_sx': (0.001, None)}
            m_exp_sx = fit_generic_model(roi_x, roi_y, err_y, gauss_exp_sx_retta_model, guess_exp_sx, lim_exp_sx)
            dict_modelli[m_exp_sx] = (gauss_exp_sx_retta_model, "Gaussiana + Coda SX + Retta")

            guess_exp_dx = {'a': ampiezza_gauss, 'mu': picco_ch, 'sigma': 10, 'c': fondo_medio, 'b': pendenza_stimata, 'A_dx': ampiezza_gauss*0.05, 'k_dx': 0.05}
            lim_exp_dx = {'a': (0, None), 'sigma': (0.1, None), 'c': (0, None), 'A_dx': (0, None), 'k_dx': (0.001, None)}
            m_exp_dx = fit_generic_model(roi_x, roi_y, err_y, gauss_exp_dx_retta_model, guess_exp_dx, lim_exp_dx)
            dict_modelli[m_exp_dx] = (gauss_exp_dx_retta_model, "Gaussiana + Coda DX + Retta")

        # SFIDE INDIPENDENTI E ANNIDATE
        # Creiamo una lista SOLO per i modelli aggiuntivi che battono la base
        candidati_extra_validi = []
        
        print("\n[TEST 2A: Aggiunta Gradino Compton]")
        if lfu.confronta_modelli_ftest(best_base, m_step, alpha=0.05, nome_picco=nomi_picchi[idx]) == m_step:
            candidati_extra_validi.append(m_step)
        
        print("\n[TEST 2B: Aggiunta Coda Esponenziale SX]")
        if lfu.confronta_modelli_ftest(best_base, m_exp_sx, alpha=0.05, nome_picco=nomi_picchi[idx]) == m_exp_sx:
            candidati_extra_validi.append(m_exp_sx)
            
        print("\n[TEST 2C: Aggiunta Coda Esponenziale DX]")
        if lfu.confronta_modelli_ftest(best_base, m_exp_dx, alpha=0.05, nome_picco=nomi_picchi[idx]) == m_exp_dx:
            candidati_extra_validi.append(m_exp_dx)
        
        # FINALISSIMA: Risoluzione logica (con Chi2 ridotto)
        if len(candidati_extra_validi) == 0:
            # Nessun F-test è stato superato dai modelli extra
            best_overall = best_base
            print("\n-> Nessun componente extra migliora significativamente la base. Vince la BASE.")
            
        elif len(candidati_extra_validi) == 1:
            # Solo un modello extra ha battuto la base, vince per direttissima
            best_overall = candidati_extra_validi[0]
            print(f"\n-> Nessun conflitto. VINCE l'aggiunta: {dict_modelli[best_overall][1]}!")
            
        else:
            # Più di un modello extra ha battuto la base: non essendo annidati tra loro,
            # lo spareggio si fa col Chi2 ridotto ESCLUDENDO la base ormai sconfitta.
            print("\n[SPAREGGIO FINALE: Confronto Modelli Aggiuntivi Validi]")
            for cand in candidati_extra_validi:
                print(f"- {dict_modelli[cand][1]}: Chi2/ndof = {cand.fval/cand.ndof:.2f}")
            
            # Vince chi ha il rapporto Chi2 / ndof minore
            best_overall = min(candidati_extra_validi, key=lambda m: m.fval / m.ndof)
            print(f"-> Allo spareggio VINCE: {dict_modelli[best_overall][1]}!")
            
        # Recupero la funzione reale e il suo nome tramite il dizionario
        func_winner, name_winner = dict_modelli[best_overall]
            
        best_models.append((best_overall, func_winner))
        best_names.append(name_winner)
        roi_data.append((roi_x, roi_y, err_y))
        
        # 4. Stampa Risultati del Vincitore
        c, c_err = best_overall.values['mu'], best_overall.errors['mu']
        s, s_err = best_overall.values['sigma'], best_overall.errors['sigma']
        
        # Risoluzione intrinseca analitica (valida SOLO per la componente gaussiana pura)
        fwhm_gauss = 2 * np.sqrt(2 * np.log(2)) * s
        fwhm_gauss_err = 2 * np.sqrt(2 * np.log(2)) * s_err
        
        # Risoluzione EFFETTIVA (calcolo numerico + errore propagato via Monte Carlo)
        # Ignoriamo il fit totale in output (usiamo `_`) perché nel plot usiamo la libreria lfu
        fwhm_effettiva, fwhm_effettiva_err = calcola_fwhm_effettiva(best_overall, roi_x)
        
        p_value = chi2.sf(best_overall.fval, best_overall.ndof) * 100
        
        print(f"\n--- RISULTATI FINALI {nomi_picchi[idx]} ---")
        print(f"Modello Selezionato : {name_winner}")
        print(f"Centroide           : {c:.2f} ± {c_err:.2f} ch")
        print(f"FWHM (solo gauss)   : {fwhm_gauss:.10f} ± {fwhm_gauss_err:.10f} ch (Sottostima la larghezza reale se c'è coda)")
        print(f"FWHM (effettiva)    : {fwhm_effettiva:.10f} ± {fwhm_effettiva_err:.10f} ch <-- VALORE FISICO DA USARE")
        print(f"Chi2/ndof           : {best_overall.fval:.1f} / {best_overall.ndof} = {best_overall.fval/best_overall.ndof:.2f}")
        print(f"p-value             : {p_value:.4f} %")

    
    # =========================================================
    # 5. PLOT DEI MODELLI VINCITORI 
    # =========================================================
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.1)
    
    for i in range(2):
        ax_main = fig.add_subplot(gs[0, i])
        ax_res = fig.add_subplot(gs[1, i], sharex=ax_main)
        
        m_fit, model_func = best_models[i]
        roi_x, roi_y, err_y = roi_data[i]
        
        # Plot generale (Fit Totale e Residui) richiamato dalla libreria esterna
        lfu.plot_fit_with_residuals(ax_main, ax_res, model_func, roi_x, roi_y, 0, err_y, m_fit, f"{nomi_picchi[i]} ({best_names[i]})")
        
        # --- PREPARAZIONE DELLE COMPONENTI GRAFICHE ---
        # Creiamo un asse X continuo per tracciare curve fluide (500 punti sono sufficienti)
        x_dense = np.linspace(roi_x[0], roi_x[-1], 500) 
        v = m_fit.values.to_dict()
        
        # Ricalcoliamo la funzione di fit totale sull'asse continuo
        fit_totale = model_func(x_dense, *m_fit.values)
        
        # 1. LA GAUSSIANA PURA:
        # A prescindere dal modello vincente, estraiamo sempre l'equazione della gaussiana
        # usando i parametri 'a', 'mu' e 'sigma'. Questo ci permette di vedere a occhio 
        # la risoluzione "ideale" senza code.
        gauss_only = v['a'] * np.exp(-(x_dense - v['mu'])**2 / (2 * v['sigma']**2))
        ax_main.plot(x_dense, gauss_only, 'b--', alpha=0.6, linewidth=2, label='Componente Gaussiana')
        
        # 2. IL BACKGROUND COMPOSITO (Grafico):
        # Per far combaciare la visualizzazione con i classici standard grafici, definiamo
        # come "Background" tutto ciò che nel fit totale NON è la componente gaussiana pura.
        # In questo modo, che il modello vincente includa una retta, un gradino o una coda, 
        # verranno tutti raggruppati magicamente nella curva verde inferiore.
        background_grafico = fit_totale - gauss_only
        ax_main.plot(x_dense, background_grafico, 'g-', alpha=0.8, linewidth=2, label='Background Totale')
        
        ax_main.legend()

    plt.show()


#_________________________________________________________________________________________________

# SINGOLA ANALISI CON MODELLO (per Sodio22 con HPGe):
# =========================================================
# MODELLO 1: GAUSSIANA + COSTANTE
# =========================================================

def gauss_costante_model(x, a, mu, sigma, c):
    """ Modello base: Gaussiana + Fondo Costante """
    gauss = a * np.exp(-(x - mu)**2 / (2 * sigma**2))
    return gauss + c

def fit_gauss_costante(roi_x, roi_y, err_y, guess_params):

    cost = LeastSquares(roi_x, roi_y, err_y, gauss_costante_model)

    m = Minuit(cost,
               guess_params['a'],
               guess_params['mu'],
               guess_params['sigma'],
               guess_params['c'])

    m.limits['a'] = (0, None)
    m.limits['sigma'] = (0.1, None)
    m.limits['c'] = (0, None)

    m.migrad()
    m.hesse()
    return m

def gauss_cost_analisi_sodio22(data, channels, window1=55, window2=65, n_first_channels_to_ignore=200, prominence=200, distance=500):
    # istogramma
    lfu.istogramma(data, channels)

    # ricerca picchi, definizione ROI e errori (poissoniani)
    picchi_ch = lfu.auto_ricerca_n_picchi(data, 2, n_first_channels_to_ignore, prominence, distance)
    picco_1_ch = picchi_ch[0]
    picco_2_ch = picchi_ch[1]

    roi_511_x = channels[picco_1_ch - window1 : picco_1_ch + window1]
    roi_511_y = data[picco_1_ch - window1 : picco_1_ch + window1]
    err_roi_511_y = np.sqrt(roi_511_y)
    err_roi_511_y[err_roi_511_y == 0] = 1.0

    roi_1274_x = channels[picco_2_ch - window2 : picco_2_ch + window2]
    roi_1274_y = data[picco_2_ch - window2 : picco_2_ch + window2]
    err_roi_1274_y = np.sqrt(roi_1274_y)
    err_roi_1274_y[err_roi_1274_y == 0] = 1.0
    
    # Fit
    guess_511 = {'a': data[picco_1_ch], 'mu': picco_1_ch, 'sigma': 10, 'c': data[picco_1_ch - window1]}
    m_511 = fit_gauss_costante(roi_511_x, roi_511_y, err_roi_511_y, guess_511)
    
    guess_1274 = {'a': data[picco_2_ch], 'mu': picco_2_ch, 'sigma': 15, 'c': data[picco_2_ch - window2]}
    m_1274 = fit_gauss_costante(roi_1274_x, roi_1274_y, err_roi_1274_y, guess_1274)
    
    # Estrazione risultati
    c511, c511_err = m_511.values['mu'], m_511.errors['mu']
    s511, s511_err = m_511.values['sigma'], m_511.errors['sigma']
    c1274, c1274_err = m_1274.values['mu'], m_1274.errors['mu']
    s1274, s1274_err = m_1274.values['sigma'], m_1274.errors['sigma']

    fwhm_511_ch = 2 * np.sqrt(2 * np.log(2)) * s511
    fwhm_511_ch_err = 2 * np.sqrt(2 * np.log(2)) * s511_err
    chi2_511_observed = m_511.fval  
    p_value_511 = chi2.sf(chi2_511_observed, m_511.ndof) 
    p_value_511_percentage = p_value_511 * 100

    fwhm_1274_ch = 2 * np.sqrt(2 * np.log(2)) * s1274
    fwhm_1274_ch_err = 2 * np.sqrt(2 * np.log(2)) * s1274_err
    chi2_1274_observed = m_1274.fval 
    p_value_1274 = chi2.sf(chi2_1274_observed, m_1274.ndof)
    p_value_1274_percentage = p_value_1274 * 100
    
    # Stampa dei risultati
    print("="*60)
    print(" RISULTATI ANALISI SODIO22 CON HPGe (Modello: GAUSSIANA + COSTANTE):")
    print("="*60)
    print(f"--- Picco 511 keV ---")
    print(f"Centroide : {c511:.2f} ± {c511_err:.2f} ch")
    print(f"FWHM (gaussiana) : {fwhm_511_ch:.10f} ± {fwhm_511_ch_err:.10f} ch")
    print(f"Chi2/ndof : {m_511.fval:.1f} / {m_511.ndof} = {m_511.fval/m_511.ndof:.2f}")
    print(f"p-value : {p_value_511_percentage:.10f} %")

    print(f"\n--- Picco 1274.5 keV ---")
    print(f"Centroide : {c1274:.2f} ± {c1274_err:.2f} ch")
    print(f"FWHM (gaussiana): {fwhm_1274_ch:.10f} ± {fwhm_1274_ch_err:.10f} ch")
    print(f"Chi2/ndof : {m_1274.fval:.1f} / {m_1274.ndof} = {m_1274.fval/m_1274.ndof:.2f}")
    print(f"p-value : {p_value_1274_percentage:.10f} %")
    print("="*60 + "\n")
    
    # Plot con analisi dei residui
    fig = plt.figure(figsize=(16, 8), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[3, 1], hspace=0.1)
    
    ax1_main = fig.add_subplot(gs[0, 0])
    ax1_res = fig.add_subplot(gs[1, 0], sharex=ax1_main)
    ax2_main = fig.add_subplot(gs[0, 1])
    ax2_res = fig.add_subplot(gs[1, 1], sharex=ax2_main)

    lfu.plot_fit_with_residuals(ax1_main, ax1_res, gauss_costante_model, roi_511_x, roi_511_y, 0, err_roi_511_y, m_511, 'Annichilazione 511 keV')
    lfu.plot_fit_with_residuals(ax2_main, ax2_res, gauss_costante_model, roi_1274_x, roi_1274_y, 0, err_roi_1274_y, m_1274, 'Decadimento 1274.5 keV')
    
    # Plotto la componente di background per i due picchi
    m_list = [m_511, m_1274]
    ax_main_list = [ax1_main, ax2_main]
    roi_511_x_dense = np.linspace(roi_511_x[0], roi_511_x[-1], 500) # Valori densi per una curva fluida
    roi_1274_x_dense = np.linspace(roi_1274_x[0], roi_1274_x[-1], 500) # Valori densi per una curva fluida
    roi_x_dense_list = [roi_511_x_dense, roi_1274_x_dense]
    for i in range(0, 2):
        m_fit = m_list[i]
        v = m_fit.values
        # Plot background
        bg_components = np.full_like(roi_x_dense_list[i], v[3])
        ax_main_list[i].plot(roi_x_dense_list[i], bg_components, 'g-', label='Fondo Costante')
        ax_main_list[i].legend()

    plt.show()