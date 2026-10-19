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
df = pd.read_excel("esperienza2.xlsx", sheet_name= "Data")
df = df.dropna(subset=["p", "q", "sigma_p", "sigma_q"])
df = df.to_numpy(dtype = 'float64')

#definizione variabili
p = df[:, 5]

q = df[:, 6]
err_p = df[:, 8]     
err_q = df[:, 7]
p=np.delete(p,[76, 77, 78, 88] )
q=np.delete(q, [76, 77, 78, 88] )
err_q=np.delete(err_q, [76, 77, 78, 88] )
err_p=np.delete(err_p, [76, 77, 78, 88] )

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
err_R = 0.025
err_L = 0.025
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
 
 
q_attesi = np.array([model_fit.Eval(x_val) for x_val in p])
residuals_norm = (q - q_attesi) / err_qeff


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

hist.Draw()

leg2 = ROOT.TLegend(0.13, 0.80, 0.35, 0.89)
leg2.AddEntry(hist, "Residui", "f")
leg2.AddEntry(gaussian_fit, "Fit gaussiano", "l")
leg2.Draw()


c2.Draw()
c2.SaveAs("hist.png")

for i in np.arange(0,len(residuals_norm)):
    if residuals_norm[i] < -25:
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

leg3 = ROOT.TLegend(0.17, 0.80, 0.45, 0.89)
leg3.AddEntry(gr2, "Residui", "pe")
leg3.AddEntry(line_fit, "Fit lineare", "l")
leg3.Draw()

c3.Draw()
c3.SaveAs("fit_line.png")

# %%
# residui normalizzati divisi perzona di p

limiti = [232, 260, 316, 340, 460, 700, 2000]

print("zona di p      N   media   errore sulla media   std")
for i in range(len(limiti) - 1):
    sel = (p >= limiti[i]) & (p < limiti[i+1]) 
    r = residuals_norm[sel]
    print(limiti[i], "-", limiti[i+1], "  ", len(r), "  ", round(np.mean(r), 2), "  ",
          round(np.std(r) / np.sqrt(len(r)), 2), "  ", round(np.std(r), 2))

r = residuals_norm[:8]
print("primi 8 punti:", len(r), "  ", round(np.mean(r), 2), "  ",
      round(np.std(r) / np.sqrt(len(r)), 2), "  ", round(np.std(r), 2))

#  punti con residuo oltre 2 sigma
print("punti con |residuo| > 2:")
for i in range(len(p)):
    if abs(residuals_norm[i]) > 2:
        print("  indice", i, " p =", p[i], " residuo normalizzato =", round(residuals_norm[i], 1),
              " residuo [mm] =", round(q[i] - q_attesi[i], 1))




