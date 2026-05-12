'''LIBRERIA CON MODELLI, FIT E ANALISI PER SPETTROSCOPIA GAMMA (PIPELINE A 4 FASI)''' 

import numpy as np
import math
import matplotlib.pyplot as plt
import inspect
from scipy.special import erfc
from scipy.signal import find_peaks
from scipy.stats import chi2, f 
from iminuit import Minuit
from iminuit.cost import LeastSquares, NormalConstraint
import libreria_funzioni_utili as lfu

# ==========================================================================
# 1. DEFINIZIONE DEI MODELLI FISICI (componente di Picco e Background)
# ==========================================================================

# ------------------------------------------
# COMPONENTI DEL PICCO (P1, P2, P3, P4)
# ------------------------------------------

def P1(x, mu, sigma, A):
    """Pura Gaussiana (Core)"""
    return A * np.exp(-(x - mu)**2 / (2 * sigma**2))

def P2(x, mu, sigma, A, T_L, lam_L):
    """Gaussiana + Coda Sinistra (ICC)"""
    gauss = P1(x, mu, sigma, A)
    arg_exp = np.clip(lam_L * (x - mu), -250, 250) # per evitare overflow
    tail_sx = T_L * np.exp(arg_exp) * erfc((x - mu + lam_L * sigma**2) / (np.sqrt(2) * sigma))
    return gauss + tail_sx

def P3(x, mu, sigma, A, T_R, lam_R):
    """Gaussiana + Coda Destra (Pile-up)"""
    gauss = P1(x, mu, sigma, A)
    arg_exp = np.clip(-lam_R * (x - mu), -250, 250)
    tail_dx = T_R * np.exp(arg_exp) * erfc((-(x - mu) + lam_R * sigma**2) / (np.sqrt(2) * sigma))
    return gauss + tail_dx

def P4(x, mu, sigma, A, T_L, lam_L, T_R, lam_R):
    """Modello Completo (Doppia EMG): Gaussiana + Coda Sinistra (ICC) + Coda Destra (Pile-up)"""
    return P2(x, mu, sigma, A, T_L, lam_L) + P3(x, mu, sigma, A, T_R, lam_R) - P1(x, mu, sigma, A)

# ------------------------------------------
# COMPONENTI DEL BACKGROUND (B1, B2, B3, B4)
# ------------------------------------------

def B1(x, c):
    """Background Costante"""
    return np.full_like(x, c, dtype=float)

def B2(x, c, b, mu):
    """Background Lineare (Retta)"""
    return c + b * (x - mu)

def B3(x, c, A_step, mu, sigma):
    """Background Costante + Gradino Multi-Compton"""
    return c + A_step * 0.5 * erfc((x - mu) / (np.sqrt(2) * sigma))

def B4(x, c, b, A_step, mu, sigma):
    """Background Lineare + Gradino Multi-Compton"""
    return c + b * (x - mu) + A_step * 0.5 * erfc((x - mu) / (np.sqrt(2) * sigma))

# REGISTRY: Dizionario esplicito e sicuro dei modelli
MODELLI_PICCO = {
    'P1': P1,
    'P2': P2,
    'P3': P3,
    'P4': P4
}

MODELLI_BACKGROUND = {
    'B1': B1,
    'B2': B2,
    'B3': B3,
    'B4': B4
}

# =============================================================================================
# FUNZIONI HELPER PER LA COSTRUZIONE DINAMICA DEI MODELLI E PER ESEGUIRE IL MODEL TOURNAMENT
# =============================================================================================

# ----------------------------------------------
# COSTRUZIONE DINAMICA DEI MODELLI 
# ----------------------------------------------
def make_combined_model(P_name, B_name):
    """
    Funzione "Factory": unisce matematicamente un modello di Picco e uno di Background,
    fondendo in modo intelligente i loro argomenti (es. mu e sigma non vengono duplicati).
    Restituisce la funzione combinata e la lista dei suoi parametri.
    """

    if P_name not in MODELLI_PICCO or B_name not in MODELLI_BACKGROUND:
        raise ValueError(f"Modello non riconosciuto. Picco: {P_name}, Bg: {B_name}")
        
    P_func = MODELLI_PICCO[P_name]
    B_func = MODELLI_BACKGROUND[B_name]
    
    # Prende la lista dei parametri delle funzioni e restituisce la lista dei nomi (stringhe) dei parametri, 
    # saltando il primo parametro (x)
    P_args = inspect.getfullargspec(P_func).args[1:] 
    B_args = inspect.getfullargspec(B_func).args[1:]
    
    # Fonde gli argomenti mantenendo l'ordine:
    # Parte dagli argomenti del picco e aggiunge quelli del background solo se non già presenti
    all_args = P_args.copy() # fa una copia dell’oggetto P_args e la assegna alla variabile all_args
    for arg in B_args:
        if arg not in all_args:
            all_args.append(arg)
            
    # Crea dinamicamente la firma della funzione combinata necessaria per Minuit
    args_str = ", ".join(all_args) # Trasforma la lista in stringa: es. "A, mu, sigma, m, q"
    
    # Creazione del codice che definisce la funzione totale (stringa!) -> qui costruisce una funzione come testo!
    code = f"def combined_model(x, {args_str}):\n"
    code += f"    return P_func(x, {', '.join(P_args)}) + B_func(x, {', '.join(B_args)})"
    
    # Crea un dizionario che definisce l’ambiente in cui verrà eseguito il codice generato (con numpy e erfc, usati nei modelli)
    local_dict = {'P_func': P_func, 'B_func': B_func, 'np': np, 'erfc': erfc}
    # Esegue il codice scritto come stringa dentro l’ambiente local_dict: è metaprogramming (codice che genera codice!!) 
    exec(code, local_dict)
    # Restituisce: la funzione combinata e la lista dei parametri
    return local_dict['combined_model'], all_args


# ----------------------------------------------
# CALCOLO CHI^2 REALE (senza penalità)
# ----------------------------------------------
def get_pure_chi2(m, x, y, err, model_func):
    """Calcola il Chi2 reale senza penalità del Soft Constraints (necessario per F-test rigoroso)"""
    sig = inspect.getfullargspec(model_func).args[1:]
    args = [m.values[k] for k in sig]
    y_fit = model_func(x, *args)
    return np.sum(((y - y_fit) / err)**2)


# ----------------------------------------------
# F-TEST DI FISHER 
# ----------------------------------------------
def is_significant(chi2_simple, ndof_simple, chi2_complex, ndof_complex, alpha=0.05):
    """Esegue l'F-test tra due modelli annidati, restituendo (Boolean, p_value)"""
    """alpha: livello di significatività (default 5% = 0.05)"""
    delta_chi2 = chi2_simple - chi2_complex
    delta_ndof = ndof_simple - ndof_complex
    
    if delta_chi2 <= 0 or delta_ndof <= 0:
        return False, 1.0 # Modello complesso ha fallito o è identico
        
    # Formula della statistica F
    f_stat = (delta_chi2 / delta_ndof) / (chi2_complex / ndof_complex)

    # Calcolo del p-value (survival function = 1 - cdf)
    p_value = f.sf(f_stat, delta_ndof, ndof_complex)
    return p_value < alpha, p_value


# ----------------------------------------------
# FWHM NUMERICA e AREA NETTA PICCO - MONTE CARLO
# ----------------------------------------------
def calcola_fwhm_effettiva_e_area_netta_picco(m_fit, P_name, roi_x, n_samples=2000):
    """
    Estrae la FWHM numerica e l'Area netta del picco, propagando gli errori
    via Monte Carlo Parametrico.
    """
    # Asse X per FWHM (molto denso ma limitato alla ROI)
    x_dense = np.linspace(roi_x[0], roi_x[-1], 10000)
    
    v_dict = m_fit.values.to_dict()
    
    # Asse X per l'Area (largo +/- 15 sigma per catturare tutte le code)
    mu_nom = v_dict['mu']
    sigma_nom = v_dict['sigma']
    x_dense_area = np.linspace(mu_nom - 15 * sigma_nom, mu_nom + 15 * sigma_nom, 5000)

    P_func = MODELLI_PICCO[P_name] 
    
    def get_fwhm_from_params(v):
        # 1. FILTRO DI FISICITÀ: Scartiamo i tiri del Monte Carlo fisicamente impossibili
        if v.get('sigma', 1) <= 0.01 or v.get('A', 1) <= 0: 
            return np.nan
        if 'lam_L' in v and v['lam_L'] <= 0: 
            return np.nan
        if 'lam_R' in v and v['lam_R'] <= 0: 
            return np.nan
            
        sig = inspect.getfullargspec(P_func).args[1:]
        p_args = [v[k] for k in sig]
        
        segnale = P_func(x_dense, *p_args)
        
        max_y = np.max(segnale)
        half_max = max_y / 2.0
        above_hm = np.where(segnale >= half_max)[0]
        
        # 2. RETURN NaN: Se il picco è deforme, restituisce NaN (non 0!)
        if len(above_hm) < 2:
            return np.nan 
            
        return x_dense[above_hm[-1]] - x_dense[above_hm[0]]
    
    def get_area_from_params(v):
        # Stessi controlli fisici della FWHM!
        if v.get('sigma', 1) <= 0.01 or v.get('A', 1) <= 0: return np.nan
        if 'lam_L' in v and v['lam_L'] <= 0: return np.nan
        if 'lam_R' in v and v['lam_R'] <= 0: return np.nan
            
        sig = inspect.getfullargspec(P_func).args[1:]
        p_args = [v[k] for k in sig]
        
        # Genera il segnale sul dominio largo
        segnale = P_func(x_dense_area, *p_args)
        
        # Integrale numerico a prova di NumPy 2.0
        try:
            return np.trapezoid(segnale, x=x_dense_area) # per versione nuova NumPy
        except AttributeError:
            return np.trapz(segnale, x_dense_area) # per versione vecchia NumPy

    # Calcolo dei valori nominali
    fwhm_effettiva = get_fwhm_from_params(v_dict)
    area_nominale = get_area_from_params(v_dict) 
    
    fwhm_err = 0.0
    area_err = 0.0 
    
    if m_fit.valid and m_fit.covariance is not None:
        param_names = list(m_fit.parameters)
        best_vals = [m_fit.values[p] for p in param_names]
        cov_matrix = np.array(m_fit.covariance) # Forza conversione in array NumPy
        
        # MONTE CARLO:
        try:
            # Estrazione dei campioni
            random_params = np.random.multivariate_normal(best_vals, cov_matrix, n_samples)
            
            fwhms_mc = []
            aree_mc = []
            
            for pars in random_params:
                p_dict = dict(zip(param_names, pars))
                
                val_fwhm = get_fwhm_from_params(p_dict)
                val_area = get_area_from_params(p_dict) 
                
                # Salviamo le FWHM valide:
                # APPLICHIAMO UN FILTRO DI SANITÀ MENTALE (generoso, non rigido):
                # La FWHM del Monte Carlo non può essere più di 4 volte la FWHM nominale
                # o minore di un quarto di essa. Se lo è, è un outlier matematico del fit.
                if not np.isnan(val_fwhm) and (fwhm_effettiva * 0.25 < val_fwhm < fwhm_effettiva * 4.0):
                    fwhms_mc.append(val_fwhm)
                    
                # Salviamo le Aree valide (accettiamo variazioni estreme da x0.1 a x10, altrimenti sono outlier matematici)
                if not np.isnan(val_area) and (area_nominale * 0.1 < val_area < area_nominale * 10.0):
                    aree_mc.append(val_area)

            # CALCOLO ERRORE: Statistica Robusta (Percentili)

            # ERRORE FWHM
            if len(fwhms_mc) > n_samples * 0.5: # Se almeno il 50% dei campioni è sano
                # Calcoliamo i percentili a 1 sigma (15.87% e 84.13%)
                q16_f, q84_f = np.percentile(fwhms_mc, [15.865, 84.135])

                # La deviazione standard robusta è la metà della distanza tra questi quantili
                fwhm_err = (q84_f - q16_f) / 2.0

            else:
                print("  [Attenzione FWHM] Matrice di covarianza instabile, troppi fallimenti nel MC.")
                fwhm_err = 0.0 # O un valore di fallback
                
            # ERRORE AREA
            if len(aree_mc) > n_samples * 0.5:
                q16_a, q84_a = np.percentile(aree_mc, [15.865, 84.135])
                area_err = (q84_a - q16_a) / 2.0
            else:
                print("  [Attenzione AREA] Matrice di covarianza instabile, troppi fallimenti nel MC.")

        except Exception as e:
            print(f"  [Errore MC] Impossibile eseguire MC: {e}")

    return fwhm_effettiva, fwhm_err, area_nominale, area_err



# ====================================================================================================
# IL MODEL TOURNAMENT (A 4 FASI) IN GENERALE (per qualsiasi sorgente, con numero arbitrario di picchi)
# ====================================================================================================

def model_tournament_analisi_generale(data, channels, nome_sorgente, energie_picchi_keV, err_energie_picchi_keV, window_list, n_first_channels_to_ignore=200, prominence=200, distance=500, verbose=True, show_plots=True, solo_picco=None, max_channel=None):
    """
    Esegue la pipeline di analisi su uno spettro gamma per un numero arbitrario di fotopicchi.
    
    PARAMETRI IN INPUT:
    - data: Lista/array con i conteggi nei vari canali 
    - channels: Lista/array con i numeri dei canali
    - nome_sorgente: (str) Es. 'Torio-228'
    - energie_picchi_keV: (list) Lista delle energie nominali dei picchi (es. [238.6, 583.2, 2614.5])
    - err_energie_picchi_keV: (list) Lista delle corrispondenti incertezze 
    - window_list: (list) Lista delle ROI window corrispondenti a ciascun picco. Deve avere la stessa lunghezza di energie_picchi_keV.
    
    Parametri in input per lfu.auto_ricerca_n_picchi:
    - n_first_channels_to_ignore: numero dei primi canali da ignorare nella ricerca dei fotopicchi
    - prominence: altezza minima per considerare un picco come tale
    - distance: minima distanza tra 2 fotopicchi successivi affinchè vengano considerati
    - max_channel: massimo canale da considerare nella ricerca dei picchi
    
    - Verbose = True: mostra le scritte di output ("verboso, dettagliato, prolisso")
    - show_plots = True: mostra i plot di output
    - solo_picco = None: studia entrambi i picchi; 
                 = 0: studia solo il picco 1, 
                 = 1: studia solo il picco 2
    
                 
    RISULTATI IN OUTPUT: le seguenti liste:
    - energie in keV dei picchi, ordinata in ordine crescente (e lista delle incertezze)
    - FWHM dei picchi (e lista delle incertezze)
    - Centroidi mu dei picchi (e lista delle incertezze)
    - FWHM relative percentuali dei picchi (e lista delle incertezze)
    - Aree dei picchi (e lista delle incertezze)
    """

    # Converte le liste in np.array (se non lo sono già)
    data = np.asarray(data, dtype=float)
    channels = np.asarray(channels, dtype=float)

    if show_plots: 
        lfu.istogramma(data[:-1], channels[:-1]) # non plotta l'ultimo canale (che raccoglie i conteggi a energie maggiori)
    
    # Estraiamo il numero totale di picchi attesi dalla lista delle energie
    N_picchi = len(energie_picchi_keV)
    
    # Controllo di sicurezza sulle dimensioni degli array in input
    if len(window_list) != N_picchi or len(err_energie_picchi_keV) != N_picchi:
        raise ValueError(f"Tutte le liste (window_list, err_energie_picchi_keV) devono avere la stessa lunghezza di energie_picchi_keV ({N_picchi}).")
    
    # Ordiniamo automaticamente le liste in base all'energia crescente (dal più piccolo al più grande)
    # zip unisce le liste a tris, sorted le ordina basandosi sul primo elemento (l'energia), e poi le separiamo di nuovo
    liste_ordinate = sorted(zip(energie_picchi_keV, err_energie_picchi_keV, window_list))
    energie_picchi_keV = [tris[0] for tris in liste_ordinate]
    err_energie_picchi_keV = [tris[1] for tris in liste_ordinate]
    window_list = [tris[2] for tris in liste_ordinate]
    
    # Il numero di picchi da cercare è 'N_picchi'
    picchi_ch = lfu.auto_ricerca_n_picchi(data, N_picchi, n_first_channels_to_ignore, prominence, distance, verbose, max_channel)
    
    # Generazione dinamica dei nomi dei picchi per i print e i plot
    nomi_picchi = [f"Picco {int(E)} keV ({nome_sorgente})" for E in energie_picchi_keV]

    best_models = []
    roi_data = []
    
    # Liste per i risultati di output
    out_mu_list, out_err_mu_list = [], []               
    out_fwhm_list, out_err_fwhm_list = [], []           
    out_fwhm_rel_list, out_err_fwhm_rel_list = [], [] 
    out_area_list, out_err_area_list = [], [] 

    # Dizionari per convertire le sigle in nomi leggibili per l'utente
    nomi_descrittivi_P = {
        'P1': 'Gaussiana Pura',
        'P2': 'Gaussiana + Coda SX',
        'P3': 'Gaussiana + Coda DX',
        'P4': 'Gaussiana + Coda SX + Coda DX'
    }
    nomi_descrittivi_B = {
        'B1': 'Costante',
        'B2': 'Retta',
        'B3': 'Costante + Gradino',
        'B4': 'Retta + Gradino'
    }

    for idx, picco_ch in enumerate(picchi_ch):

        #-------------------------------------
        # --- BLOCCO DI BYPASS: se vogliamo studiare solo uno dei due picchi ---
        # Se abbiamo chiesto di analizzare un solo picco, e questo indice non è quello richiesto, lo saltiamo
        if solo_picco is not None and idx != solo_picco:
            # Riempiamo le liste con dei 'NaN' (Not a Number) per mantenere intatta la lunghezza delle liste (indice 0 = Picco1, indice 1 = Picco2)
            out_fwhm_list.append(np.nan)
            out_err_fwhm_list.append(np.nan)
            out_mu_list.append(np.nan)
            out_err_mu_list.append(np.nan)
            out_fwhm_rel_list.append(np.nan)
            out_err_fwhm_rel_list.append(np.nan)
            out_area_list.append(np.nan)      
            out_err_area_list.append(np.nan)  
            roi_data.append((None, None, None))
            best_models.append((None, None, None, None))
            continue # passa subito al prossimo giro del ciclo for
        # -------------------------------------

        if verbose: print(f"\n\n{'='*70}\n PIPELINE PICCO: {nomi_picchi[idx]} (Canale ~{picco_ch})\n{'='*70}")
        
        # La window viene presa dalla lista fornita dall'utente tramite l'indice 'idx'
        current_window = window_list[idx]
        roi_x = channels[int(picco_ch - current_window) : int(picco_ch + current_window)]
        roi_y = data[int(picco_ch - current_window) : int(picco_ch + current_window)]
        err_y = np.sqrt(roi_y) # incertezza poissoniana
        err_y[err_y == 0] = 1.0
        
        # -------------------------------------------------------------------
        # FASE 1: PRE-FIT (Inizializzazione Robusta) -> per stimare mu e sigma (in modo preliminare)
        # -------------------------------------------------------------------
        if verbose: print("[FASE 1] Esecuzione Pre-Fit (Gaussiana + Retta)...") 
        # guess per il pre-fit
        fondo_sx, fondo_dx = np.mean(roi_y[:5]), np.mean(roi_y[-5:])
        fondo_medio = (fondo_sx + fondo_dx) / 2
        A_stimata = max(10, data[picco_ch] - fondo_medio)
        
        comb_pre, _ = make_combined_model('P1', 'B2') # GAUSSIANA + RETTA (prefit semplice)
        cost_pre = LeastSquares(roi_x, roi_y, err_y, comb_pre)
        m_pre = Minuit(cost_pre, mu=picco_ch, sigma=10, A=A_stimata, c=fondo_medio, b=0)
        m_pre.limits['sigma'] = (0.1, None); m_pre.limits['A'] = (0, None)
        m_pre.migrad()
        
        mu_pre = m_pre.values['mu']
        sig_pre = m_pre.values['sigma']
        
        # -------------------------------------------------------------------
        # FASE 2: TORNEO DEL BACKGROUND (BIC-test su ROI Mascherata)
        # -------------------------------------------------------------------
        if verbose: print("[FASE 2] Torneo del Background (con Mascheramento del picco)...")
        # Maschera simmetrica per evitare inquinamento dalle code ICC (e di pile-up, ma meno probabile allontanando sorgente)
        # Attenzione: la ROI deve essere sufficientemente grande!
        mask = (roi_x < mu_pre - 4 * sig_pre) | (roi_x > mu_pre + 4 * sig_pre)
        x_mask, y_mask, err_mask = roi_x[mask], roi_y[mask], err_y[mask]
        
        b_results = {}
        for B_name in ['B1', 'B2', 'B3', 'B4']:

            if B_name not in MODELLI_BACKGROUND:
                raise ValueError(f"Modello non riconosciuto. Bg: {B_name}")
        
            B_func = MODELLI_BACKGROUND[B_name]
            cost_B = LeastSquares(x_mask, y_mask, err_mask, B_func)
            
            # Costruzione dinamica del dizionario dei parametri iniziali
            guess_B = {'c': fondo_medio}
            if 'b' in inspect.getfullargspec(B_func).args: guess_B['b'] = 0.0
            if 'A_step' in inspect.getfullargspec(B_func).args: guess_B['A_step'] = max(0, fondo_sx - fondo_dx)
            if 'mu' in inspect.getfullargspec(B_func).args: guess_B['mu'] = mu_pre
            if 'sigma' in inspect.getfullargspec(B_func).args: guess_B['sigma'] = sig_pre
            
            m_B = Minuit(cost_B, **guess_B)
            # Fissiamo i parametri legati al core del picco per non instabilizzare l'erfc
            if 'mu' in m_B.parameters: m_B.fixed['mu'] = True
            if 'sigma' in m_B.parameters: m_B.fixed['sigma'] = True
            m_B.migrad()
            
            # BIC-test (Bayesian Information Criterion test)
            k = m_B.nfit # Numero di parametri liberi
            bic = m_B.fval + k * np.log(len(x_mask))
            b_results[B_name] = {'minuit': m_B, 'bic': bic, 'k': k}
            if verbose: print(f"  - {B_name}: BIC = {bic:.2f} (Parametri liberi: {k})")
            
        # Seleziona l'elemento di b_results con il valore 'bic' più basso 
        # Restituisce il nome del modello migliore secondo il criterio BIC
        best_B_name = min(b_results, key=lambda k: b_results[k]['bic']) # (la chiave per il confronto è bic)
        b_opt_minuit = b_results[best_B_name]['minuit']
        if verbose: print(f"  => VINCITORE BACKGROUND (Minimo BIC): {best_B_name}")
        
        # -------------------------------------------------------------------
        # FASE 3: TORNEO DEL PICCO (F-Test con Soft Constraints)
        # -------------------------------------------------------------------
        if verbose: print("\n[FASE 3] Torneo del Picco (Soft Constraints sul Background)...")
        b_opt_vals = b_opt_minuit.values.to_dict()
        b_opt_errs = b_opt_minuit.errors.to_dict()
        
        # Prepariamo la penalità Gaussiana (Priors) per i parametri del background
        p_names, p_vals, p_errs = [], [], [] # p sta per "parametri" del background
        for k in b_opt_vals:
            if k not in ['mu', 'sigma']: # mu e sigma ora devono essere liberi!
                p_names.append(k); p_vals.append(b_opt_vals[k]); p_errs.append(b_opt_errs[k])
                
        p_results = {}
        for P_name in ['P1', 'P2', 'P3', 'P4']:
            comb_func, _ = make_combined_model(P_name, best_B_name)
            lsq = LeastSquares(roi_x, roi_y, err_y, comb_func)
            
            # Funzione di costo con SOFT CONSTRAINTS (vincoli morbidi per c, b, A_step):
            # 1. Costruisce la funzione di costo totale come somma del chi² dei dati e di una penalità gaussiana 
            #    per ogni parametro del background (esclusi mu e sigma, liberi di variare)
            # 2. La penalità (NormalConstraint -> che è per es. (c - c_opt)^2 / err_c_opt^2) vincola i parametri del background ai valori già stimati
            #    I parametri non sono fissati: possono variare, ma allontanarsi costa in termini di chi² (evita che il background si adatti troppo al picco falsando il fit)
            #    (È un compromesso tra parametri liberi e parametri completamente fissati)
            cost_tot = lsq + NormalConstraint(p_names, p_vals, p_errs) if p_names else lsq
            
            guess_P = {'mu': mu_pre, 'sigma': sig_pre, 'A': A_stimata}
            # Inizializzazioni conservative per le code (fondamentale!) 
            if P_name in ['P2', 'P4']: guess_P.update({'T_L': A_stimata*0.05, 'lam_L': 1.0/(3*sig_pre)})
            if P_name in ['P3', 'P4']: guess_P.update({'T_R': A_stimata*0.05, 'lam_R': 1.0/(3*sig_pre)})
            for k in p_names: guess_P[k] = b_opt_vals[k] # Init background
                
            m_P = Minuit(cost_tot, **guess_P)

            # --- LIMITI FISICI RIGOROSI (Previene la degenerazione delle EMG) ---
            # 1. Il "cuore" gaussiano DEVE dominare. Non può morire (min 50%) e non può esplodere.
            m_P.limits['A'] = (A_stimata * 0.5, A_stimata * 1.5) 
            
            # 2. La risoluzione (sigma) deve restare vicina a quella del pre-fit.
            m_P.limits['sigma'] = (sig_pre * 0.6, sig_pre * 1.5) 
            
            # 3. Le code EMG (difetti) NON possono superare il 30% dell'altezza del picco!
            if 'T_L' in m_P.parameters: 
                m_P.limits['T_L'] = (0.0, A_stimata * 0.30)
            if 'T_R' in m_P.parameters: 
                m_P.limits['T_R'] = (0.0, A_stimata * 0.30)
                
            # 4. Il parametro di decadimento lambda deve avere senso fisico.
            # Se lambda è troppo piccolo la coda è piatta, se è troppo grande è un picco a delta.
            # Lo costringiamo in un range realistico legato alla sigma del rivelatore.
            if 'lam_L' in m_P.parameters: 
                m_P.limits['lam_L'] = (1.0 / (15 * sig_pre), 1.0 / (0.5 * sig_pre))
            if 'lam_R' in m_P.parameters: 
                m_P.limits['lam_R'] = (1.0 / (15 * sig_pre), 1.0 / (0.5 * sig_pre))
            # --------------------------------------------------------------------
            
            m_P.migrad()
            
            # Calcolo del vero Chi2 (esclusa la penalità del background) per l'F-Test
            pure_chi2 = get_pure_chi2(m_P, roi_x, roi_y, err_y, comb_func)
            ndof_real = len(roi_x) - m_P.nfit 
            p_results[P_name] = {'minuit': m_P, 'chi2': pure_chi2, 'ndof': ndof_real}

        # LOGICA DEL TORNEO (F-Test)
        chi2_P1, ndof_P1 = p_results['P1']['chi2'], p_results['P1']['ndof']
        sig_L, p_L = is_significant(chi2_P1, ndof_P1, p_results['P2']['chi2'], p_results['P2']['ndof'])
        sig_R, p_R = is_significant(chi2_P1, ndof_P1, p_results['P3']['chi2'], p_results['P3']['ndof'])
        
        if verbose:
            print(f"  - Test Coda ICC (P2 vs P1): p-value = {p_L*100:.4f} % -> {'SIGNIFICATIVA' if sig_L else 'Trascurabile'}")
            print(f"  - Test Coda Pile-up (P3 vs P1): p-value = {p_R*100:.4f} % -> {'SIGNIFICATIVO' if sig_R else 'Trascurabile'}")
        
        if not sig_L and not sig_R:
            best_P_name = 'P1'
        elif sig_L and not sig_R:
            best_P_name = 'P2'
        elif sig_R and not sig_L:
            best_P_name = 'P3'
        else:
            # Entrambe le singole code migliorano la base. P4 giustifica l'uso di ENTRAMBE?
            sig_4vs2, _ = is_significant(p_results['P2']['chi2'], p_results['P2']['ndof'], p_results['P4']['chi2'], p_results['P4']['ndof'])
            sig_4vs3, _ = is_significant(p_results['P3']['chi2'], p_results['P3']['ndof'], p_results['P4']['chi2'], p_results['P4']['ndof'])
            if sig_4vs2 and sig_4vs3:
                best_P_name = 'P4'
            else:
                # Se P4 non vince, prendi la coda singola con il miglior Chi2 ridotto
                best_P_name = 'P2' if (p_results['P2']['chi2']/p_results['P2']['ndof'] < p_results['P3']['chi2']/p_results['P3']['ndof']) else 'P3'
                
        if verbose: print(f"  => VINCITORE PICCO: {best_P_name}")
        
        # -------------------------------------------------------------------
        # FASE 4: GLOBAL REFIT di rifinitura (Matrice Hessiana Finale)
        # -------------------------------------------------------------------
        if verbose: print("\n[FASE 4] Fit Globale di Rifinitura...")
        comb_func, _ = make_combined_model(best_P_name, best_B_name)
        lsq_final = LeastSquares(roi_x, roi_y, err_y, comb_func) # NESSUNA penalità
        
        m_final = Minuit(lsq_final, **p_results[best_P_name]['minuit'].values.to_dict())
        m_final.limits = p_results[best_P_name]['minuit'].limits # Ripristiniamo i limiti definiti prima
        
        m_final.migrad()
        m_final.hesse() # Estrazione esatta della Matrice di Covarianza
        
        # -----------------------------------------------------
        # ESTRAZIONE RISULTATI
        #---------------------
        # Salvataggio dati per il plot
        best_models.append((m_final, comb_func, best_P_name, best_B_name))
        roi_data.append((roi_x, roi_y, err_y))

        # Chi^2 per valutare bontà fit
        chi2_final = m_final.fval
        ndof_final = len(roi_x) - m_final.nfit
        # Calcolo del p-value (Survival Function del Chi2)
        p_value_fit = chi2.sf(chi2_final, ndof_final)
        
        # Estrazione numerica FWHM effettiva e Area Netta (con Monte Carlo)
        fwhm_eff, fwhm_eff_err, area_netta, area_netta_err = calcola_fwhm_effettiva_e_area_netta_picco(m_final, best_P_name, roi_x)
        
        out_fwhm_list.append(fwhm_eff)
        out_err_fwhm_list.append(fwhm_eff_err)
        out_area_list.append(area_netta)
        out_err_area_list.append(area_netta_err)
        
        # Centroide
        c, c_err = m_final.values['mu'], m_final.errors['mu']
                
        # Risoluzione relativa percentuale:
        # Calcolo del rapporto FWHM/Centroide*100 (risoluzione relativa PERCENTUALE) e propagazione dell'errore
        # supponendo fwhm_eff e centroide scorrelati (corretto, poiché fwhm_eff calcolata in modo numerico)
        fwhm_relativa = (fwhm_eff / c)*100
        fwhm_relativa_err = fwhm_relativa * np.sqrt((fwhm_eff_err / fwhm_eff)**2 + (c_err / c)**2)
        
        # Salvataggio nelle nuove liste per il return finale
        out_mu_list.append(c)
        out_err_mu_list.append(c_err)
        out_fwhm_rel_list.append(fwhm_relativa)
        out_err_fwhm_rel_list.append(fwhm_relativa_err)

        # Calcolo FWHM solo dal core gaussiano (2.355 * sigma), per rapido confronto con la FWHM effettiva trovata
        sigma_gauss = m_final.values['sigma']
        sigma_gauss_err = m_final.errors['sigma']
        fwhm_gaussiana = 2.35482 * sigma_gauss
        fwhm_gaussiana_err = 2.35482 * sigma_gauss_err
        
        # -----------------------------------------------------
        # STAMPA A SCHERMO
        #-----------------
        if verbose:
            print(f"\n--- RISULTATI DEFINITIVI {nomi_picchi[idx]} ---")
            # Uso i dizionari creati prima per stampare i nomi espliciti
            print(f"Modello Combinato       : {nomi_descrittivi_P[best_P_name]} + {nomi_descrittivi_B[best_B_name]}")
            print(f"Centroide               : {c:.2f} ± {c_err:.2f} ch")
            print(f"FWHM (da sigma)         : {fwhm_gaussiana:.4f} ± {fwhm_gaussiana_err:.4f} ch")
            print(f"FWHM (effettiva)        : {fwhm_eff:.4f} ± {fwhm_eff_err:.4f} ch")
            print(f"FWHM (eff. relativa %)  : {fwhm_relativa:.5f} ± {fwhm_relativa_err:.5f} %")
            print(f"Area Netta              : {area_netta:.1f} ± {area_netta_err:.1f} conteggi") # <--- NUOVO
            print(f"Chi2 / ndof             : {chi2_final:.1f} / {ndof_final} = {chi2_final/ndof_final:.2f}")
            print(f"p-value del fit         : {p_value_fit*100:.4f} %") 
        # -----------------------------------------------------
        
    # ====================================================================
    # 5. PLOT DEI MODELLI VINCITORI 
    # ====================================================================
    if show_plots:
        # Logica per creare una griglia dinamica in base al numero di picchi.
        # Imposta max 2 colonne. Se ho 3 picchi, farà 2 righe e 2 colonne (lasciando l'ultimo vuoto).
        n_cols = min(2, N_picchi)
        n_rows = math.ceil(N_picchi / n_cols)
        
        # Dimensione dinamica della figura (altezza proporzionale al numero di righe)
        fig = plt.figure(figsize=(16 if n_cols == 2 else 8, 8 * n_rows), constrained_layout=True)
        
        # Gridspec esterno per posizionare ogni blocco di analisi
        outer_gs = fig.add_gridspec(n_rows, n_cols)
    
        for i in range(N_picchi):
           # Se un picco è stato saltato (solo_picco attivo), ignoriamo il plot per quell'indice
           if best_models[i][0] is None:
               continue
               
           row = i // n_cols
           col = i % n_cols
           
           # Usiamo un subgridspec per dividere la singola cella in (Plot Principale 75% / Residui 25%)
           inner_gs = outer_gs[row, col].subgridspec(2, 1, height_ratios=[3, 1], hspace=0.05)
           
           ax_main = fig.add_subplot(inner_gs[0])
           ax_res = fig.add_subplot(inner_gs[1], sharex=ax_main)
           
           m_fit, comb_func, P_name, B_name = best_models[i]
           roi_x, roi_y, err_y = roi_data[i]
        
           titolo = f"{nomi_picchi[i]}\n({nomi_descrittivi_P[P_name]} + {nomi_descrittivi_B[B_name]})"
           
           lfu.plot_fit_with_residuals(ax_main, ax_res, comb_func, roi_x, roi_y, 0, err_y, m_fit, titolo)
        
           x_dense = np.linspace(roi_x[0], roi_x[-1], 500) 
           v = m_fit.values.to_dict()
        
           gauss_only = P1(x_dense, v['mu'], v['sigma'], v['A'])
           fit_totale = comb_func(x_dense, *m_fit.values)
           tailing_e_background_grafico = fit_totale - gauss_only
        
           ax_main.plot(x_dense, gauss_only, 'b--', alpha=0.6, linewidth=2, label='Cuore Gaussiano')
           ax_main.plot(x_dense, tailing_e_background_grafico, 'g-', alpha=0.8, linewidth=2, label='Tailing + Background')
           ax_main.legend()

        plt.show()

    return energie_picchi_keV, err_energie_picchi_keV, out_fwhm_list, out_err_fwhm_list, out_mu_list, out_err_mu_list, out_fwhm_rel_list, out_err_fwhm_rel_list, out_area_list, out_err_area_list


# ================================================================================
# STUDIO DELLA WINDOW (per eventuale errore sistematico) IN GENERALE
# (cioè per qualsiasi sorgente)
# ================================================================================

def studio_sistematico_window_generale(data, channels, nome_sorgente, energie_picchi_keV, err_energie_picchi_keV, picco_idx, window_range, window_list_base, n_first_channels_to_ignore=2000, prominence=50, distance=1000):
    """
    Esegue uno studio di sensibilità sulla Risoluzione Relativa (%) variando la semilarghezza 
    della ROI (window) di un solo picco specifico di una determinata sorgente.
    
    Parametri:
    - data, channels: i dati dello spettro.
    - nome_sorgente, energie_picchi_keV, err_energie_picchi_keV: Definiscono la sorgente analizzata.
    - picco_idx: L'indice nella lista 'energie_picchi_keV' del picco che stiamo analizzando (0, 1, 2...).
    - window_range: array con i valori della window da testare sul picco scelto (es. np.arange(50, 120, 5)),
       importante: devo scegliere un range in cui la window è "buona" (secondo i criteri messi in latex)
    - window_list_base: (list) Le window standard per TUTTI i picchi (es. [50, 70, 80]). 
                        Quella all'indice 'picco_idx' verrà ignorata e sostituita iterativamente.
    - altri: parametri per la ricerca automatica dei picchi.
    
    Ritorna: fwhm_rel_media, errore_sistematico
    """
    
    fwhm_rel_variabili = []
    err_stat_fwhm_rel = []
    
    # Nomi dinamici basati sull'indice richiesto
    energia_scelta = energie_picchi_keV[picco_idx]
    nome_picco = f"Picco a {energia_scelta} keV ({nome_sorgente})"
    
    print(f"Inizio test di sensibilità per il {nome_picco}...")
    print(f"Variando la Window | Range: da {window_range[0]} a {window_range[-1]} canali")
    
    for w_var in window_range:
        # Creiamo una copia per non modificare la lista base originale durante il ciclo
        current_windows = window_list_base.copy()
        
        # Sostituiamo SOLO la window del picco in esame con il valore del range corrente
        current_windows[picco_idx] = w_var
        
        # Chiamata silenziosa alla funzione principale generale
        analisi = model_tournament_analisi_generale(
            data, channels, nome_sorgente, energie_picchi_keV, err_energie_picchi_keV,
            window_list=current_windows, 
            n_first_channels_to_ignore=n_first_channels_to_ignore, 
            prominence=prominence, distance=distance, 
            verbose=False, show_plots=False,
            solo_picco=picco_idx 
        )
        
        # Estraiamo i valori - Indici attuali di 'analisi':
        # 0: energie_picchi_keV,  1: err_energie_picchi_keV
        # 2: fwhm,  3: err_fwhm,  4: mu,  5: err_mu
        # 6: fwhm_rel,  7: err_fwhm_rel
        # 8: area_netta,  9: err_area_netta
        
        fwhm_relativa = analisi[6][picco_idx]
        errore_stat_relativo = analisi[7][picco_idx]
        
        fwhm_rel_variabili.append(fwhm_relativa)
        err_stat_fwhm_rel.append(errore_stat_relativo)

    fwhm_rel_variabili = np.array(fwhm_rel_variabili)
    err_stat_fwhm_rel = np.array(err_stat_fwhm_rel)

    # Calcolo Statistico
    errore_sistematico = np.std(fwhm_rel_variabili, ddof=1)
    fwhm_rel_media = np.mean(fwhm_rel_variabili)

    print(f"\n--- RISULTATI STUDIO SISTEMATICO ---")
    print(f"Risoluzione Relativa Media : {fwhm_rel_media:.7f} %")
    print(f"Errore Sistematico         : {errore_sistematico:.7f} %")

    # Plot dei risultati
    plt.figure(figsize=(10, 6))
    plt.errorbar(window_range, fwhm_rel_variabili, yerr=err_stat_fwhm_rel, fmt='o-', color='black', capsize=5, label='Risoluzione Relativa (%)')
    
    plt.axhline(fwhm_rel_media, color='red', linestyle='--', label=f'Media ({fwhm_rel_media:.4f} %)')
    plt.fill_between(window_range, fwhm_rel_media - errore_sistematico, fwhm_rel_media + errore_sistematico, 
                     color='red', alpha=0.2, label=f'$\\sigma_{{sys}}$ ($\\pm${errore_sistematico:.4f} %)')

    plt.title(f'Sensibilità della Risoluzione Relativa alla Window della ROI\n{nome_picco}')
    plt.xlabel('Window [canali]')
    plt.ylabel('Risoluzione Relativa (%)')
    plt.legend()
    plt.grid(True, linestyle=':', alpha=0.7)
    plt.tight_layout()
    plt.show()

    return fwhm_rel_media, errore_sistematico
