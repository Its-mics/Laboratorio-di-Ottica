# %%
import ROOT
import numpy as np
import pandas as pd
 
ROOT.gStyle.SetOptFit(1111)    # aggiunge il label con chi2/ndf e probabilita e parametri se metti 0000 toglie i dati

#funzione modello
def model(x, par):
    p = x[0]
    A = par[0]
    B = par[1]
    C = par[2]
    return B + (C*(p-A))/((p-A)-C)

def model_derivative(p, par):
    return -par[2]**2 / ((p - par[0]) - par[2])**2

#leggere file excel 
xlsx ="esperienza2.xlsx"
df = pd.read_excel("esperienza2.xlsx", sheet_name= "Dati")
df = df.dropna(subset=["p", "q", "sigma_p", "sigma_q"])
df = df.to_numpy(dtype = 'float64')

#definizione variabili
p = df[:, 5]
q = df[:, 6]

err_p = df[:, 8]

err_p = df[:, 8]
err_q = df[:, 7]

p = np.delete(p, [0, 74, 75, 76, 77, 88] )
q = np.delete(q, [0, 74, 75, 76, 77, 88]  )
err_p = np.delete(err_p, [0, 74, 75, 76, 77, 88] )
err_q = np.delete(err_q, [0, 74, 75, 76, 77, 88] )


#minimizzazzione con ROOT
model_fit = ROOT.TF1('model', model, min(p), max(p), 3)
model_fit.SetParameters(0.0, 0.0, 0.0)
model_fit.SetParNames("A", "B", "C")

gr1 = ROOT.TGraphErrors(len(p), p, q, err_p, err_q)
gr1.SetTitle("Dati sperimentali e fit; p; q")
gr1.SetMarkerStyle(20)
gr1.SetMarkerSize(0.7)                    
gr1.SetMarkerColor(ROOT.kAzure + 2)
gr1.SetLineColor(ROOT.kAzure + 3)        
gr1.SetLineWidth(1)


fit_res = gr1.Fit(model_fit, "SR")

fitStatus = fit_res.Status()
print(f"Fit status: {fitStatus}")

par = model_fit.GetParameters()

errors = model_fit.GetParErrors()

chi2 = fit_res.Chi2()

err_qeff = np.sqrt(err_q**2 + ((model_derivative(p, par)) * err_p)**2) #errore effettivo

print("Parametri stimati:")
print(f"A = {par[0]} ± {errors[0]}")
print(f"B = {par[1]} ± {errors[1]}")
print(f"C = {par[2]} ± {errors[2]}")
print(f"Chi2 = {chi2}")

c1 = ROOT.TCanvas("c1", "Fit con ROOT", 800, 600)
c1.SetGrid()

gr1.Draw("AP")


leg1 = ROOT.TLegend(0.17, 0.80, 0.45, 0.89)
leg1.AddEntry(gr1, "Dati", "pe")
leg1.AddEntry(model_fit, "Fit", "l")
leg1.Draw()

c1.Draw()
c1.SaveAs("fit_root.png")

# %%
err_R = 1
err_L = 1
err_x0 = 0.115470053837925

if np.abs(par[0]) < 3*(errors[0] + err_x0 + err_R + err_L):
        print("A è compatibile con 0")
else :
        print("A NON è compatibile con 0")

print("---")

if np.abs(par[1]) < 3*(errors[1] + err_x0  + err_L):
    print("B è compatibile con 0")
else :
    print("B NON è compatibile con 0")
 
 


# %%
q_attesi = np.array([model_fit.Eval(x_val) for x_val in p])
residuals_norm = (q - q_attesi) / err_qeff


#residuals_norm = np.delete(residuals_norm, [0,1, 10, 12, 14, 43, 44, 52])


gaussian_fit = ROOT.TF1('gaussian', 'gaus', min(q), max(q), 3)
gaussian_fit.SetParameters(1, 0, 1)
gaussian_fit.SetParNames("const", "mu", "sigma")
gaussian_fit.SetLineColor(ROOT.kRed + 1)
gaussian_fit.SetLineWidth(3)

hist = ROOT.TH1F("hist", "Istogramma dei dati; x; y", 20, min(residuals_norm), max(residuals_norm))
for val in residuals_norm:
    hist.Fill(val)

hist.SetLineColor(ROOT.kAzure + 3)
hist.SetLineWidth(2)
hist.SetFillColorAlpha(ROOT.kAzure + 1, 0.55)   
hist.GetXaxis().SetTitle("residuo normalizzato")
hist.GetYaxis().SetTitle("conteggi")
hist.SetTitle("Istogramma residui normalizzati")

r = hist.Fit(gaussian_fit, "S")

fitStatus = r.Status()
print(f"Fit status: {fitStatus}")

parameters = gaussian_fit.GetParameters()
errors = gaussian_fit.GetParErrors()

chi2 = gaussian_fit.GetChisquare()

print("Parametri stimati:")
print(f"const = {parameters[0]} ± {errors[0]}")
print(f"#mu = {parameters[1]} ± {errors[1]}")
print(f"#sigma = {parameters[2]} ± {errors[2]}")
print(f"Chi2 = {chi2}")


c2 = ROOT.TCanvas("c2", "Istogramma dei dati", 800, 600)
c2.SetGrid()

ROOT.gStyle.SetStatX(0.50)
ROOT.gStyle.SetStatY(0.70)

hist.Draw()

leg2 = ROOT.TLegend(0.80, 0.40, 0.89, 0.50)
leg2.AddEntry(hist, "Residui", "f")
leg2.AddEntry(gaussian_fit, "Fit gaussiano", "l")
leg2.Draw()


c2.Draw()
c2.SaveAs("hist.png")


print()
print()

for i in np.arange(0,len(residuals_norm)):
    if residuals_norm[i] < -20:
        print (residuals_norm[i])
        print(i)

# %%
if np.abs(parameters[0] - 1) < 3*errors[0]:
    print("const è compatibile con 1")
else :
    print("const NON è compatibile con 1")

print("---")

if np.abs(parameters[1]) < 3*errors[1]:
    print("mu è compatibile con 0")
else :
    print("mu NON è compatibile con 0")

print("---")

if np.abs(parameters[2] - 1) < 3*errors[2]:
    print("sigma è compatibile con 1")
else :
    print("sigma NON è compatibile con 1")

# %%
def line(x, par):
    x = x[0]
    m = par[0]
    q = par[1]
    return m * x + q

residuals = q - q_attesi
print(residuals)

line_fit = ROOT.TF1('line', line, min(p), max(p), 2)
line_fit.SetParameters(1, 0)
line_fit.SetParNames("m", "q")
line_fit.SetLineColor(ROOT.kRed + 1)
line_fit.SetLineWidth(2)

gr2 = ROOT.TGraphErrors(len(p), p, residuals, np.zeros_like(residuals), err_qeff)
gr2.SetTitle("Residui in funzione di p; p; q - q_{att}")
gr2.SetMarkerStyle(20)
gr2.SetMarkerSize(0.7)
gr2.SetMarkerColor(ROOT.kAzure + 2)
gr2.SetLineColor(ROOT.kAzure + 3)

fit_res = gr2.Fit(line_fit, "S")

fitStatus = fit_res.Status()
print(f"Fit status: {fitStatus}")

parameters = line_fit.GetParameters()
errors = line_fit.GetParErrors()

chi2 = fit_res.Chi2()

print("Parametri stimati:")
print(f"m = {parameters[0]} ± {errors[0]}")
print(f"q = {parameters[1]} ± {errors[1]}")
print(f"Chi2 = {chi2}")

c3 = ROOT.TCanvas("c3", "Fit con ROOT", 800, 600)
c3.SetGrid()

gr2.Draw("AP")

ROOT.gStyle.SetStatX(0.70)
ROOT.gStyle.SetStatY(0.70)

leg3 = ROOT.TLegend(0.17, 0.80, 0.45, 0.89)
leg3.AddEntry(gr2, "Residui", "pe")
leg3.AddEntry(line_fit, "Fit lineare", "l")
leg3.Draw()

c3.Draw()
c3.SaveAs("fit_line.png")

# %%
# Scarto verticale di ogni punto rispetto alla retta fittata
scarti_retta = residuals - np.array([line_fit.Eval(float(x)) for x in p])

# Seleziona i due punti più sotto la retta
indici_sotto = np.flatnonzero(scarti_retta < 0)
indici_evidenziati = [61, 87]


# Grafico sovrapposto con i due punti evidenziati
gr_evidenziati = ROOT.TGraphErrors(
    len(indici_evidenziati),
    np.asarray(p[indici_evidenziati], dtype="d"),
    np.asarray(residuals[indici_evidenziati], dtype="d"),
    np.zeros(len(indici_evidenziati), dtype="d"),
    np.asarray(err_qeff[indici_evidenziati], dtype="d")
)
gr_evidenziati.SetMarkerStyle(24)  # cerchio vuoto
gr_evidenziati.SetMarkerSize(2)
gr_evidenziati.SetMarkerColor(ROOT.kRed)
gr_evidenziati.SetLineColor(ROOT.kRed)


for i in indici_evidenziati:
    print(
        f"Punto evidenziato: p = {p[i]:.2f}, "
        f"residuo = {residuals[i]:.2f}, "
        f"scarto dalla retta = {scarti_retta[i]:.2f}"
    )


gr2.Draw("AP")
gr_evidenziati.Draw("P SAME")



leg3 = ROOT.TLegend(0.17, 0.80, 0.48, 0.91)
leg3.AddEntry(gr2, "Residui", "pe")
leg3.AddEntry(line_fit, "Fit lineare", "l")
leg3.AddEntry(gr_evidenziati, "Punti sotto la retta", "p")
leg3.Draw()


c3.SaveAs("zoomcaz.png")
c3.Draw()

# %%
# residui normalizzati divisi perzona di p





