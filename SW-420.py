import sys
import serial
import serial.tools.list_ports
import numpy as np
import csv
import json
import math
import os
from datetime import datetime
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QComboBox, QFileDialog, QStatusBar,
    QGroupBox, QSplitter, QFrame, QSizePolicy
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QColor, QPalette
import pyqtgraph as pg


# ─────────────────────────────────────────────────────────
#  CONSTANTES
# ─────────────────────────────────────────────────────────
BUFFER_SIZE      = 512
SAMPLE_RATE_HZ   = 1000
ADC_MAX          = 4095
THRESHOLD        = 1500

# La FFT solo se dibuja si la energía RMS de la señal supera este valor.
# Por debajo = señal plana / ruido → se muestra pantalla "sin actividad".
FFT_MIN_RMS      = 30          # ADC counts  (ajusta según tu ruido real)

# Recalcular FFT cada N muestras nuevas (4 = muy fluido, 8 = menos CPU)
FFT_REFRESH_EVERY = 4

PALETTE = {
    "bg":       "#07090f",
    "panel":    "#101620",
    "border":   "#1e2d3d",
    "accent":   "#00e5ff",
    "accent2":  "#ff6b35",
    "ok":       "#39ff14",
    "warn":     "#ffcc00",
    "danger":   "#ff2d55",
    "text":     "#c9d1d9",
    "text_dim": "#586069",
    "plot_bg":  "#090e18",
    "html":     "#a44bf8",
}

STYLE_MAIN = f"""
    QMainWindow, QWidget {{
        background-color: {PALETTE['bg']};
        color: {PALETTE['text']};
        font-family: 'Consolas', 'Courier New', monospace;
    }}
    QGroupBox {{
        border: 1px solid {PALETTE['border']};
        border-radius: 6px;
        margin-top: 10px;
        padding-top: 6px;
        font-size: 11px;
        color: {PALETTE['text_dim']};
        letter-spacing: 2px;
        text-transform: uppercase;
    }}
    QGroupBox::title {{ subcontrol-origin: margin; left: 10px; top: -1px; }}
    QComboBox {{
        background-color: {PALETTE['panel']};
        border: 1px solid {PALETTE['border']};
        border-radius: 4px;
        color: {PALETTE['accent']};
        padding: 5px 10px;
        font-family: 'Consolas', monospace;
        font-size: 12px;
        min-height: 28px;
    }}
    QComboBox::drop-down {{ border: none; width: 28px; }}
    QComboBox QAbstractItemView {{
        background-color: {PALETTE['panel']};
        border: 1px solid {PALETTE['border']};
        color: {PALETTE['text']};
        selection-background-color: {PALETTE['border']};
    }}
    QStatusBar {{
        background-color: {PALETTE['panel']};
        border-top: 1px solid {PALETTE['border']};
        color: {PALETTE['text_dim']};
        font-size: 11px;
        padding: 2px 8px;
    }}
    QSplitter::handle {{ background: {PALETTE['border']}; width: 1px; height: 1px; }}
    QScrollBar:vertical {{ background: {PALETTE['panel']}; width: 6px; }}
    QScrollBar::handle:vertical {{ background: {PALETTE['border']}; border-radius: 3px; }}
"""

BTN_PRIMARY = f"""
    QPushButton {{
        background-color: transparent; border: 1px solid {PALETTE['accent']};
        border-radius: 5px; color: {PALETTE['accent']};
        font-family: 'Consolas', monospace; font-size: 12px;
        font-weight: bold; letter-spacing: 1px; padding: 7px 18px; min-height: 32px;
    }}
    QPushButton:hover {{ background-color: rgba(0,229,255,0.10); }}
    QPushButton:pressed {{ background-color: rgba(0,229,255,0.20); }}
    QPushButton:disabled {{ border-color: {PALETTE['text_dim']}; color: {PALETTE['text_dim']}; }}
"""
BTN_DANGER = f"""
    QPushButton {{
        background-color: transparent; border: 1px solid {PALETTE['danger']};
        border-radius: 5px; color: {PALETTE['danger']};
        font-family: 'Consolas', monospace; font-size: 12px;
        font-weight: bold; letter-spacing: 1px; padding: 7px 18px; min-height: 32px;
    }}
    QPushButton:hover {{ background-color: rgba(255,45,85,0.10); }}
    QPushButton:disabled {{ border-color: {PALETTE['text_dim']}; color: {PALETTE['text_dim']}; }}
"""
BTN_OK = f"""
    QPushButton {{
        background-color: transparent; border: 1px solid {PALETTE['ok']};
        border-radius: 5px; color: {PALETTE['ok']};
        font-family: 'Consolas', monospace; font-size: 12px;
        font-weight: bold; letter-spacing: 1px; padding: 7px 18px; min-height: 32px;
    }}
    QPushButton:hover {{ background-color: rgba(57,255,20,0.10); }}
    QPushButton:disabled {{ border-color: {PALETTE['text_dim']}; color: {PALETTE['text_dim']}; }}
"""
BTN_HTML = f"""
    QPushButton {{
        background-color: transparent; border: 1px solid {PALETTE['html']};
        border-radius: 5px; color: {PALETTE['html']};
        font-family: 'Consolas', monospace; font-size: 12px;
        font-weight: bold; letter-spacing: 1px; padding: 7px 18px; min-height: 32px;
    }}
    QPushButton:hover {{ background-color: rgba(168,85,247,0.12); border-color: #c084fc; color: #c084fc; }}
    QPushButton:pressed {{ background-color: rgba(168,85,247,0.22); }}
    QPushButton:disabled {{ border-color: {PALETTE['text_dim']}; color: {PALETTE['text_dim']}; }}
"""


# ─────────────────────────────────────────────────────────
#  HILO SERIAL
# ─────────────────────────────────────────────────────────
class SerialReader(QThread):
    data_received = pyqtSignal(float)
    error_signal  = pyqtSignal(str)

    def __init__(self, port, baudrate=115200):
        super().__init__()
        self.port = port; self.baudrate = baudrate
        self._running = True; self.ser = None

    def run(self):
        try:
            self.ser = serial.Serial(self.port, self.baudrate, timeout=1)
        except Exception as e:
            self.error_signal.emit(str(e)); return
        while self._running:
            try:
                line = self.ser.readline().decode(errors='replace').strip()
                if line:
                    self.data_received.emit(float(line))
            except ValueError:
                pass
            except Exception as e:
                if self._running:
                    self.error_signal.emit(str(e))
                break

    def stop(self):
        self._running = False
        if self.ser and self.ser.is_open:
            self.ser.close()
        self.wait(1000)


# ─────────────────────────────────────────────────────────
#  REPORTE HTML
# ─────────────────────────────────────────────────────────
class HTMLReportGenerator:
    def __init__(self, csv_rows, puerto, baud):
        self.rows = csv_rows; self.puerto = puerto; self.baud = baud
        self.ts_captura = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    def _analizar(self):
        if not self.rows: return None
        vals = [r[2] for r in self.rows]; ts = [r[1] for r in self.rows]; n = len(vals)
        v_max = max(vals); v_min = min(vals); v_mean = sum(vals)/n
        v_std = math.sqrt(sum((v-v_mean)**2 for v in vals)/n)
        dur_ms = ts[-1]-ts[0] if n>1 else 0
        if n>1:
            ints = [ts[i+1]-ts[i] for i in range(min(200,n-1)) if ts[i+1]>ts[i]]
            dt_ms = sum(ints)/len(ints) if ints else 10
        else: dt_ms = 10
        fs = 1000/dt_ms
        N_fft = min(512,n); arr = np.array(vals[:N_fft],dtype=np.float64)
        arr -= arr.mean(); arr *= np.hanning(N_fft)
        fft_mag = np.abs(np.fft.rfft(arr)); freqs = np.fft.rfftfreq(N_fft,d=1.0/fs)
        freqs = freqs[1:]; fft_mag = fft_mag[1:]
        dom_idx = int(np.argmax(fft_mag)); freq_dom = float(freqs[dom_idx])
        sorted_idx = np.argsort(fft_mag)[::-1][:5]
        top5 = [(float(freqs[i]),float(fft_mag[i])) for i in sorted_idx]
        eventos = []; en_ev = False; e_ini = 0; e_vals = []
        for i,v in enumerate(vals):
            if v > THRESHOLD:
                if not en_ev: en_ev=True; e_ini=i; e_vals=[]
                e_vals.append(v)
            else:
                if en_ev and len(e_vals)>=3:
                    eventos.append({"ts_ini":ts[e_ini],"ts_fin":ts[i-1],"dur_ms":ts[i-1]-ts[e_ini],"pico":max(e_vals),"media":sum(e_vals)/len(e_vals),"n":len(e_vals)})
                en_ev=False; e_vals=[]
        if en_ev and len(e_vals)>=3:
            eventos.append({"ts_ini":ts[e_ini],"ts_fin":ts[-1],"dur_ms":ts[-1]-ts[e_ini],"pico":max(e_vals),"media":sum(e_vals)/len(e_vals),"n":len(e_vals)})
        rango = ADC_MAX/10
        hist_labels = [f"{int(b*rango)}–{int((b+1)*rango)-1}" for b in range(10)]
        hist_vals = [sum(1 for v in vals if int(b*rango)<=v<=int((b+1)*rango)-1) for b in range(10)]
        step = max(1,n//600)
        return {"n":n,"ts":ts[::step],"vals":vals[::step],"v_max":v_max,"v_min":v_min,
                "v_mean":round(v_mean,2),"v_std":round(v_std,2),"dur_ms":dur_ms,"dur_s":round(dur_ms/1000,2),
                "fs":round(fs,2),"dt_ms":round(dt_ms,2),"freq_dom":round(freq_dom,4),
                "freqs_fft":[round(float(f),4) for f in freqs[:128]],"mags_fft":[round(float(m),2) for m in fft_mag[:128]],
                "top5":top5,"eventos":eventos,"hist_labels":hist_labels,"hist_vals":hist_vals}

    def generar(self, path):
        r = self._analizar()
        if not r: return False
        sat_pct = round(r["v_max"]/ADC_MAX*100,1)
        snr_db = round(20*math.log10(r["v_max"]/r["v_std"]) if r["v_std"]>0 else 0,1)
        n_ev = len(r["eventos"])
        ev_rows = ""
        for i,e in enumerate(r["eventos"],1):
            dur_s=e["dur_ms"]/1000
            nivel="CRÍTICO" if e["pico"]>ADC_MAX*0.8 else "ALTO" if e["pico"]>ADC_MAX*0.5 else "MODERADO"
            cls={"CRÍTICO":"critico","ALTO":"alto","MODERADO":"moderado"}[nivel]
            tipo="Impulso primario" if i==1 else "Rebote / eco mecánico" if e["dur_ms"]<500 else "Vibración secundaria"
            ev_rows+=f'<tr><td><span class="ev-badge">EVT {i:02d}</span></td><td>{e["ts_ini"]:,} ms</td><td>{e["ts_fin"]:,} ms</td><td>{dur_s:.3f} s</td><td>{e["pico"]:,} ADC</td><td>{e["media"]:.1f} ADC</td><td>{e["n"]:,}</td><td><span class="nivel-badge {cls}">{nivel}</span></td><td>{tipo}</td></tr>'
        mag_max = r["top5"][0][1] if r["top5"] else 1
        freq_rows = ""
        for i,(f,m) in enumerate(r["top5"],1):
            pct=m/mag_max*100 if mag_max>0 else 0; per=round(1000/f,1) if f>0 else 0
            freq_rows+=f'<tr><td>#{i}</td><td>{f:.4f} Hz</td><td>{per} ms</td><td><div class="bar-wrap"><div class="bar-fill" style="width:{pct:.1f}%"></div></div><span class="bar-val">{m:.1f}</span></td></tr>'
        # espejo FFT para el reporte HTML también
        neg_freqs = [-f for f in reversed(r["freqs_fft"])]
        neg_mags  = list(reversed(r["mags_fft"]))
        c_freqs   = neg_freqs + r["freqs_fft"]
        c_mags    = neg_mags  + r["mags_fft"]
        html = f"""<!DOCTYPE html><html lang="es"><head><meta charset="UTF-8"/>
<title>Vibration Report · {self.ts_captura}</title>
<script src="https://cdnjs.cloudflare.com/ajax/libs/Chart.js/4.4.1/chart.umd.min.js"></script>
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Syne:wght@400;600;800&display=swap');
:root{{--bg:#0b0f1a;--surface:#111827;--surface2:#1a2236;--border:#1f2e45;--accent:#00d4ff;--accent2:#7c3aed;--warn:#f59e0b;--danger:#ef4444;--ok:#10b981;--text:#e2e8f0;--muted:#64748b;--mono:'Space Mono',monospace;--sans:'Syne',sans-serif;}}
*{{box-sizing:border-box;margin:0;padding:0}}body{{background:var(--bg);color:var(--text);font-family:var(--sans)}}
.wrap{{max-width:1300px;margin:0 auto;padding:2rem 1.5rem 4rem}}
.header{{display:flex;justify-content:space-between;flex-wrap:wrap;gap:1.5rem;border-bottom:1px solid var(--border);padding-bottom:2rem;margin-bottom:2.5rem}}
.header h1{{font-size:clamp(1.5rem,4vw,2.4rem);font-weight:800;background:linear-gradient(135deg,var(--accent),var(--accent2));-webkit-background-clip:text;-webkit-text-fill-color:transparent}}
.header p{{color:var(--muted);font-family:var(--mono);font-size:.76rem;margin-top:.5rem}}
.badge{{background:var(--surface2);border:1px solid var(--border);border-radius:8px;padding:.75rem 1.25rem;font-family:var(--mono);font-size:.74rem;color:var(--accent);text-align:right;line-height:2}}
.pills{{display:flex;flex-wrap:wrap;gap:.5rem;margin-bottom:2rem}}
.pill{{background:var(--surface2);border:1px solid var(--border);border-radius:999px;padding:.3rem .9rem;font-family:var(--mono);font-size:.7rem;color:var(--muted)}}.pill span{{color:var(--accent)}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(175px,1fr));gap:1rem;margin-bottom:2.5rem}}
.card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.2rem 1.4rem}}
.card:hover{{border-color:var(--accent)}}
.lbl{{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--muted);margin-bottom:.35rem}}
.val{{font-family:var(--mono);font-size:1.65rem;font-weight:700;color:var(--accent)}}
.unit{{font-size:.72rem;color:var(--muted);margin-top:.3rem;font-family:var(--mono)}}
.sec{{margin-bottom:2.5rem}}
.sec-title{{font-size:.67rem;letter-spacing:.2em;text-transform:uppercase;color:var(--accent);font-family:var(--mono);margin-bottom:1rem;display:flex;align-items:center;gap:.75rem}}
.sec-title::after{{content:'';flex:1;height:1px;background:var(--border)}}
.chart-card{{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:1.5rem}}
.two-col{{display:grid;grid-template-columns:1fr 1fr;gap:1.5rem}}
@media(max-width:860px){{.two-col{{grid-template-columns:1fr}}}}
table{{width:100%;border-collapse:collapse;font-family:var(--mono);font-size:.77rem}}
thead th{{background:var(--surface2);color:var(--muted);font-size:.64rem;letter-spacing:.1em;text-transform:uppercase;padding:.6rem 1rem;text-align:left;border-bottom:1px solid var(--border)}}
tbody tr{{border-bottom:1px solid var(--border)}}tbody tr:hover{{background:var(--surface2)}}tbody td{{padding:.72rem 1rem}}
.ev-badge{{background:linear-gradient(135deg,var(--accent2),#4f46e5);color:#fff;font-size:.63rem;font-weight:700;padding:.2em .6em;border-radius:4px}}
.nivel-badge{{font-size:.63rem;font-weight:700;padding:.2em .7em;border-radius:4px}}
.critico{{background:rgba(239,68,68,.15);color:var(--danger);border:1px solid rgba(239,68,68,.3)}}
.alto{{background:rgba(245,158,11,.15);color:var(--warn);border:1px solid rgba(245,158,11,.3)}}
.moderado{{background:rgba(16,185,129,.15);color:var(--ok);border:1px solid rgba(16,185,129,.3)}}
.bar-wrap{{display:inline-block;width:90px;height:5px;background:var(--border);border-radius:3px;vertical-align:middle;margin-right:.5rem}}
.bar-fill{{height:100%;background:linear-gradient(90deg,var(--accent),var(--accent2));border-radius:3px}}
.bar-val{{color:var(--muted);font-size:.7rem}}
.interp-grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(270px,1fr));gap:1rem}}
.interp-card{{background:var(--surface);border:1px solid var(--border);border-radius:10px;padding:1.2rem 1.4rem}}
.interp-card h4{{font-size:.68rem;letter-spacing:.12em;text-transform:uppercase;color:var(--accent2);margin-bottom:.45rem}}
.interp-card p{{font-size:.81rem;color:var(--muted);line-height:1.65}}.interp-card strong{{color:var(--text)}}
.footer{{margin-top:3rem;border-top:1px solid var(--border);padding-top:1.5rem;text-align:center;font-family:var(--mono);font-size:.68rem;color:var(--muted)}}.footer strong{{color:var(--accent)}}
</style></head><body><div class="wrap">
<header class="header"><div><h1>Vibration Analysis Report</h1><p>SW-420 · ESP32 · ADC 12-bit · {self.ts_captura}</p></div>
<div class="badge">MUESTRAS&nbsp;&nbsp;{r['n']:,}<br>DURACIÓN&nbsp;&nbsp;{r['dur_s']} s<br>Fs&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;{r['fs']} Hz<br>EVENTOS&nbsp;&nbsp;&nbsp;{n_ev}</div></header>
<div class="pills"><div class="pill">Puerto <span>{self.puerto}</span></div><div class="pill">Baudrate <span>{self.baud}</span></div><div class="pill">ADC máx <span>{ADC_MAX}</span></div><div class="pill">Umbral <span>{THRESHOLD} ADC</span></div><div class="pill">Δt <span>{r['dt_ms']} ms</span></div></div>
<section class="sec"><div class="sec-title">Estadísticas globales</div><div class="grid">
<div class="card"><div class="lbl">Amplitud máxima</div><div class="val">{r['v_max']:,}</div><div class="unit">ADC · {sat_pct}% fondo escala</div></div>
<div class="card"><div class="lbl">Amplitud mínima</div><div class="val">{r['v_min']}</div><div class="unit">ADC counts</div></div>
<div class="card"><div class="lbl">Media</div><div class="val">{r['v_mean']}</div><div class="unit">ADC counts</div></div>
<div class="card"><div class="lbl">Desv. estándar σ</div><div class="val">{r['v_std']}</div><div class="unit">ADC counts</div></div>
<div class="card"><div class="lbl">Duración</div><div class="val">{r['dur_s']}</div><div class="unit">s · {r['dur_ms']:,} ms</div></div>
<div class="card"><div class="lbl">Fs real</div><div class="val">{r['fs']}</div><div class="unit">Hz · Δt={r['dt_ms']} ms</div></div>
<div class="card"><div class="lbl">Frec. dominante</div><div class="val">{r['freq_dom']}</div><div class="unit">Hz (Hanning)</div></div>
<div class="card"><div class="lbl">SNR estimado</div><div class="val">{snr_db}</div><div class="unit">dB</div></div>
</div></section>
<section class="sec"><div class="sec-title">Señal temporal</div><div class="chart-card"><div style="position:relative;height:300px"><canvas id="cS"></canvas></div></div></section>
<section class="sec"><div class="sec-title">Eventos detectados — {n_ev} evento{'s' if n_ev!=1 else ''}</div><div class="chart-card"><div style="overflow-x:auto"><table><thead><tr><th>ID</th><th>Inicio</th><th>Fin</th><th>Duración</th><th>Pico ADC</th><th>Media ADC</th><th>Muestras</th><th>Nivel</th><th>Clasificación</th></tr></thead><tbody>{ev_rows or '<tr><td colspan="9" style="color:var(--muted);text-align:center;padding:1.5rem">Sin eventos detectados</td></tr>'}</tbody></table></div></div></section>
<section class="sec"><div class="sec-title">Espectro FFT centrado &amp; Histograma</div><div class="two-col">
<div class="chart-card"><div style="font-size:.6rem;letter-spacing:.15em;color:var(--accent);margin-bottom:.75rem">FFT CENTRADA EN 0 Hz</div><div style="position:relative;height:240px"><canvas id="cF"></canvas></div></div>
<div class="chart-card"><div style="font-size:.6rem;letter-spacing:.15em;color:var(--accent);margin-bottom:.75rem">HISTOGRAMA DE AMPLITUDES</div><div style="position:relative;height:240px"><canvas id="cH"></canvas></div></div>
</div></section>
<section class="sec"><div class="sec-title">Top 5 frecuencias dominantes</div><div class="chart-card"><div style="overflow-x:auto"><table><thead><tr><th>Rank</th><th>Frecuencia</th><th>Período</th><th>Magnitud relativa</th></tr></thead><tbody>{freq_rows}</tbody></table></div></div></section>
<section class="sec"><div class="sec-title">Interpretación técnica</div><div class="interp-grid">
<div class="interp-card"><h4>Morfología</h4><p>Se detectaron <strong>{n_ev} evento(s)</strong> sobre {THRESHOLD} ADC. {"Decaimiento exponencial típico de impacto." if n_ev>=1 else "Señal en nivel de ruido durante toda la captura."}</p></div>
<div class="interp-card"><h4>Saturación</h4><p>Pico máximo: <strong>{sat_pct}% del fondo de escala</strong>. {"⚠️ Riesgo de saturación." if sat_pct>85 else "✅ Margen dinámico adecuado." if sat_pct<70 else "⚡ Margen ajustado."}</p></div>
<div class="interp-card"><h4>Frec. dominante</h4><p>FFT identifica <strong>{r['freq_dom']} Hz</strong>. {"Sub-Hz: movimiento lento o resonancia estructural." if r['freq_dom']<1 else f"Período ≈ {round(1000/r['freq_dom'],1)} ms." if r['freq_dom']>0 else "Indeterminada."}</p></div>
<div class="interp-card"><h4>Calidad</h4><p><strong>{r['n']:,} muestras</strong> a <strong>{r['fs']} Hz</strong>. SNR estimado <strong>{snr_db} dB</strong>. {"✅ Excelente." if snr_db>30 else "⚠️ SNR bajo." if snr_db<15 else "✅ Aceptable."}</p></div>
</div></section>
<footer class="footer">Generado por <strong>ESP32 Sensor Pro</strong> · {self.ts_captura} · {r['n']:,} muestras</footer>
</div>
<script>
Chart.defaults.color='#64748b';Chart.defaults.borderColor='#1f2e45';Chart.defaults.font.family="'Space Mono',monospace";Chart.defaults.font.size=11;
const ts={json.dumps(r['ts'])},vals={json.dumps(r['vals'])};
const cf={json.dumps(c_freqs)},cm={json.dumps(c_mags)};
const hl={json.dumps(r['hist_labels'])},hv={json.dumps(r['hist_vals'])};
function colorByVal(v){{const r=v/{ADC_MAX};return r>0.7?'#ef4444':r>0.4?'#f59e0b':'#10b981';}}
const ctxS=document.getElementById('cS').getContext('2d');
const gS=ctxS.createLinearGradient(0,0,0,300);gS.addColorStop(0,'rgba(0,212,255,0.22)');gS.addColorStop(1,'rgba(0,212,255,0)');
new Chart(ctxS,{{type:'line',data:{{labels:ts,datasets:[{{label:'ADC',data:vals,borderColor:vals.map(colorByVal),backgroundColor:gS,borderWidth:1.3,pointRadius:0,tension:0.1,fill:true}},{{label:'Umbral',data:ts.map(()=>{THRESHOLD}),borderColor:'rgba(245,158,11,0.55)',borderWidth:1,borderDash:[6,4],pointRadius:0,fill:false}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{labels:{{font:{{size:10}},boxWidth:10}}}}}},scales:{{x:{{ticks:{{maxTicksLimit:10,callback:(v)=>ts[v]+' ms'}},grid:{{color:'#1f2e45'}}}},y:{{min:0,max:{ADC_MAX},ticks:{{maxTicksLimit:6}},grid:{{color:'#1f2e45'}}}}}}}}}});
new Chart(document.getElementById('cF').getContext('2d'),{{type:'bar',data:{{labels:cf.map(f=>f.toFixed(2)),datasets:[{{label:'Magnitud',data:cm,backgroundColor:cf.map(f=>Math.abs(f)<1?'#00d4ff':'rgba(0,212,255,0.3)'),borderWidth:0}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}},tooltip:{{callbacks:{{title:c=>c[0].label+' Hz'}}}}}},scales:{{x:{{ticks:{{maxTicksLimit:10,font:{{size:9}}}},grid:{{display:false}}}},y:{{ticks:{{maxTicksLimit:5}},grid:{{color:'#1f2e45'}}}}}}}}}});
new Chart(document.getElementById('cH').getContext('2d'),{{type:'bar',data:{{labels:hl,datasets:[{{label:'Muestras',data:hv,backgroundColor:hl.map((_,i)=>`hsla(${{160-i*16}},80%,55%,0.78)`),borderWidth:0,borderRadius:4}}]}},options:{{responsive:true,maintainAspectRatio:false,plugins:{{legend:{{display:false}}}},scales:{{x:{{ticks:{{maxTicksLimit:6,font:{{size:8}}}},grid:{{display:false}}}},y:{{ticks:{{maxTicksLimit:5}},grid:{{color:'#1f2e45'}}}}}}}}}});
</script></body></html>"""
        try:
            with open(path,"w",encoding="utf-8") as f: f.write(html)
            return True
        except: return False


# ─────────────────────────────────────────────────────────
#  VENTANA PRINCIPAL
# ─────────────────────────────────────────────────────────
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("ESP32  ·  SENSOR PRO  ·  SW-420")
        self.resize(1280, 780)
        self.setStyleSheet(STYLE_MAIN)

        self.serial_thread  = None
        self.pausado        = False
        self.grabando       = False
        self.csv_rows       = []
        self.t_inicio_grab  = None
        self.sample_index   = 0
        self._puerto_activo = ""
        self._baud_activo   = 115200
        self._fft_counter   = 0
        self._fft_active    = False   # True cuando hay señal real

        self.buf_signal = deque([0.0] * BUFFER_SIZE, maxlen=BUFFER_SIZE)
        self._build_ui()
        self._refresh_ports()

    # ── UI ────────────────────────────────────────────────
    def _build_ui(self):
        central = QWidget(); self.setCentralWidget(central)
        root = QVBoxLayout(central)
        root.setContentsMargins(12,10,12,6); root.setSpacing(8)
        root.addWidget(self._make_header())
        root.addWidget(self._make_toolbar(), 0)
        splitter = QSplitter(Qt.Vertical); splitter.setHandleWidth(4)
        splitter.addWidget(self._make_signal_panel())
        splitter.addWidget(self._make_fft_panel())
        splitter.setSizes([420,280]); root.addWidget(splitter,1)
        self.status = QStatusBar(); self.setStatusBar(self.status)
        self._set_status("Desconectado — selecciona un puerto y presiona CONECTAR","dim")

    def _make_header(self):
        frame = QFrame(); frame.setFixedHeight(64)
        frame.setStyleSheet(f"QFrame{{background:qlineargradient(x1:0,y1:0,x2:1,y2:0,stop:0 #0d1117,stop:0.5 #0f1923,stop:1 #0d1117);border-bottom:1px solid {PALETTE['border']};border-radius:6px;}}")
        hl = QHBoxLayout(frame); hl.setContentsMargins(18,0,18,0)
        title = QLabel("◈  SW-420  VIBRATION ANALYZER")
        title.setFont(QFont("Consolas",17,QFont.Bold))
        title.setStyleSheet(f"color:{PALETTE['accent']};letter-spacing:3px;")
        hl.addWidget(title); hl.addStretch()
        self.lbl_valor = QLabel("——")
        self.lbl_valor.setFont(QFont("Consolas",28,QFont.Bold))
        self.lbl_valor.setStyleSheet(f"color:{PALETTE['ok']};letter-spacing:2px;")
        self.lbl_valor.setAlignment(Qt.AlignRight|Qt.AlignVCenter)
        hl.addWidget(self.lbl_valor)
        lbl_unit = QLabel("ADC"); lbl_unit.setFont(QFont("Consolas",11))
        lbl_unit.setStyleSheet(f"color:{PALETTE['text_dim']};margin-left:4px;margin-right:8px;")
        hl.addWidget(lbl_unit)
        self.lbl_conn = QLabel("● OFFLINE")
        self.lbl_conn.setFont(QFont("Consolas",11,QFont.Bold))
        self.lbl_conn.setStyleSheet(f"color:{PALETTE['danger']};letter-spacing:2px;")
        hl.addWidget(self.lbl_conn)
        return frame

    def _make_toolbar(self):
        bar = QFrame(); bar.setFixedHeight(77)
        bar.setStyleSheet(f"QFrame{{background-color:{PALETTE['panel']};border:1px solid {PALETTE['border']};border-radius:6px;}}")
        hl = QHBoxLayout(bar); hl.setContentsMargins(12,6,12,6); hl.setSpacing(10)

        grp_port = self._labeled_group("PUERTO")
        g_lay = QHBoxLayout(grp_port); g_lay.setContentsMargins(6,0,6,0)
        self.combo_port = QComboBox(); self.combo_port.setMinimumWidth(150)
        g_lay.addWidget(self.combo_port)
        btn_r = QPushButton("⟳"); btn_r.setFixedWidth(32); btn_r.setStyleSheet(BTN_PRIMARY)
        btn_r.setToolTip("Actualizar puertos"); btn_r.clicked.connect(self._refresh_ports)
        g_lay.addWidget(btn_r); hl.addWidget(grp_port)

        grp_baud = self._labeled_group("BAUDRATE")
        gb_lay = QHBoxLayout(grp_baud); gb_lay.setContentsMargins(8,0,8,0)
        self.combo_baud = QComboBox()
        for b in ["9600","19200","57600","115200","230400","921600"]: self.combo_baud.addItem(b)
        self.combo_baud.setCurrentText("115200"); self.combo_baud.setMinimumWidth(100)
        gb_lay.addWidget(self.combo_baud); hl.addWidget(grp_baud)
        hl.addSpacing(4)

        self.btn_conectar = QPushButton("⚡  CONECTAR")
        self.btn_conectar.setStyleSheet(BTN_OK)
        self.btn_conectar.clicked.connect(self._toggle_conexion); hl.addWidget(self.btn_conectar)

        self.btn_pausa = QPushButton("⏸  PAUSAR")
        self.btn_pausa.setStyleSheet(BTN_PRIMARY); self.btn_pausa.setEnabled(False)
        self.btn_pausa.clicked.connect(self._toggle_pausa); hl.addWidget(self.btn_pausa)
        hl.addStretch()

        grp_rec = self._labeled_group("DATOS")
        gr_lay = QHBoxLayout(grp_rec); gr_lay.setContentsMargins(6,0,6,0); gr_lay.setSpacing(6)
        self.btn_record = QPushButton("⏺  GRABAR")
        self.btn_record.setStyleSheet(BTN_DANGER); self.btn_record.setEnabled(False)
        self.btn_record.clicked.connect(self._toggle_grabacion); gr_lay.addWidget(self.btn_record)
        self.btn_export_csv = QPushButton("💾  CSV")
        self.btn_export_csv.setStyleSheet(BTN_PRIMARY); self.btn_export_csv.setEnabled(False)
        self.btn_export_csv.clicked.connect(self._exportar_csv); gr_lay.addWidget(self.btn_export_csv)
        self.btn_export_html = QPushButton("📊  REPORTE HTML")
        self.btn_export_html.setStyleSheet(BTN_HTML); self.btn_export_html.setEnabled(False)
        self.btn_export_html.setToolTip("Genera reporte interactivo")
        self.btn_export_html.clicked.connect(self._exportar_html); gr_lay.addWidget(self.btn_export_html)
        hl.addWidget(grp_rec)
        return bar

    def _labeled_group(self, label):
        g = QGroupBox(label)
        g.setStyleSheet(f"QGroupBox{{font-size:9px;color:{PALETTE['text_dim']};border:1px solid {PALETTE['border']};border-radius:4px;margin-top:8px;padding-top:4px;letter-spacing:2px;}}QGroupBox::title{{subcontrol-origin:margin;left:8px;}}")
        return g

    def _make_signal_panel(self):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame{{background:{PALETTE['panel']};border:1px solid {PALETTE['border']};border-radius:6px;}}")
        lay = QVBoxLayout(frame); lay.setContentsMargins(8,6,8,6); lay.setSpacing(4)
        hdr = QHBoxLayout()
        lbl = QLabel("▸  SEÑAL EN TIEMPO REAL"); lbl.setFont(QFont("Consolas",10,QFont.Bold))
        lbl.setStyleSheet(f"color:{PALETTE['text_dim']};letter-spacing:3px;"); hdr.addWidget(lbl); hdr.addStretch()
        self.lbl_rec_dot = QLabel("●"); self.lbl_rec_dot.setStyleSheet(f"color:{PALETTE['danger']};font-size:14px;")
        self.lbl_rec_dot.setVisible(False); hdr.addWidget(self.lbl_rec_dot)
        self.lbl_rec_count = QLabel(""); self.lbl_rec_count.setFont(QFont("Consolas",10))
        self.lbl_rec_count.setStyleSheet(f"color:{PALETTE['danger']};"); hdr.addWidget(self.lbl_rec_count)
        lay.addLayout(hdr)
        pg.setConfigOptions(antialias=True, foreground=PALETTE['text_dim'])
        self.plot_signal = pg.PlotWidget(); self.plot_signal.setBackground(PALETTE['plot_bg'])
        self.plot_signal.showGrid(x=True,y=True,alpha=0.2)
        self.plot_signal.setLabel('left','Amplitud (ADC)'); self.plot_signal.setLabel('bottom',f'Muestras  [{BUFFER_SIZE} pts]')
        self.plot_signal.getAxis('left').setTextPen(PALETTE['text_dim'])
        self.plot_signal.getAxis('bottom').setTextPen(PALETTE['text_dim'])
        self.plot_signal.setYRange(0,4096); self.plot_signal.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)
        pen_signal = pg.mkPen(color=(0,229,255),width=2)
        self.curve_signal = self.plot_signal.plot(pen=pen_signal)
        self.threshold_line = pg.InfiniteLine(pos=THRESHOLD,angle=0,
            pen=pg.mkPen(color=(255,107,53),width=1,style=Qt.DashLine),
            label='umbral',labelOpts={'color':PALETTE['accent2'],'position':0.05})
        self.plot_signal.addItem(self.threshold_line)
        lay.addWidget(self.plot_signal)
        return frame

    def _make_fft_panel(self):
        frame = QFrame()
        frame.setStyleSheet(f"QFrame{{background:{PALETTE['panel']};border:1px solid {PALETTE['border']};border-radius:6px;}}")
        lay = QVBoxLayout(frame); lay.setContentsMargins(8,6,8,6); lay.setSpacing(4)
        hdr = QHBoxLayout()
        lbl = QLabel("▸  FFT — ESPECTRO CENTRADO  (−Fs/2 … 0 … +Fs/2)")
        lbl.setFont(QFont("Consolas",10,QFont.Bold))
        lbl.setStyleSheet(f"color:{PALETTE['text_dim']};letter-spacing:3px;"); hdr.addWidget(lbl); hdr.addStretch()
        self.lbl_dom_freq = QLabel("Dom. Freq: ——  Hz"); self.lbl_dom_freq.setFont(QFont("Consolas",11))
        self.lbl_dom_freq.setStyleSheet(f"color:{PALETTE['accent2']};"); hdr.addWidget(self.lbl_dom_freq)
        lay.addLayout(hdr)

        # Stack: plot FFT encima, overlay "sin actividad" encima del plot
        self.plot_fft = pg.PlotWidget(); self.plot_fft.setBackground(PALETTE['plot_bg'])
        self.plot_fft.showGrid(x=True,y=True,alpha=0.2)
        self.plot_fft.setLabel('left','|Magnitud|'); self.plot_fft.setLabel('bottom','Frecuencia (Hz)')
        self.plot_fft.getAxis('left').setTextPen(PALETTE['text_dim'])
        self.plot_fft.getAxis('bottom').setTextPen(PALETTE['text_dim'])
        self.plot_fft.setSizePolicy(QSizePolicy.Expanding,QSizePolicy.Expanding)

        # Línea de referencia en 0 Hz
        zero_line = pg.InfiniteLine(pos=0,angle=90,
            pen=pg.mkPen(color=(255,255,255,50),width=1,style=Qt.DotLine))
        self.plot_fft.addItem(zero_line)

        pen_fft = pg.mkPen(color=(255,107,53),width=2)
        self.curve_fft = self.plot_fft.plot(pen=pen_fft)
        self._fft_fill = pg.FillBetweenItem(self.curve_fft,self.plot_fft.plot([0],[0]),
            brush=pg.mkBrush(255,107,53,30))
        self.plot_fft.addItem(self._fft_fill)
        lay.addWidget(self.plot_fft)

        # Overlay "sin actividad" — se muestra cuando no hay señal
        self.fft_overlay = QLabel(
            "⬤  SIN ACTIVIDAD\n"
            "Esperando señal sobre el umbral de ruido\n"
            f"RMS mínimo requerido: {FFT_MIN_RMS} ADC"
        )
        self.fft_overlay.setAlignment(Qt.AlignCenter)
        self.fft_overlay.setFont(QFont("Consolas",12))
        self.fft_overlay.setStyleSheet(f"""
            QLabel {{
                color: {PALETTE['text_dim']};
                background: rgba(9,14,24,0.88);
                border-radius: 8px;
                letter-spacing: 2px;
                padding: 18px;
            }}
        """)
        # Lo ponemos como widget hijo flotante del plot_fft
        self.fft_overlay.setParent(self.plot_fft)
        self.fft_overlay.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.fft_overlay.show()

        lay.addWidget(self.plot_fft)
        return frame

    def resizeEvent(self, event):
        """Mantiene el overlay centrado cuando la ventana cambia de tamaño."""
        super().resizeEvent(event)
        self._reposition_overlay()

    def _reposition_overlay(self):
        if hasattr(self,'fft_overlay') and hasattr(self,'plot_fft'):
            pw = self.plot_fft.width(); ph = self.plot_fft.height()
            ow = min(500, pw-40); oh = 90
            self.fft_overlay.setFixedSize(ow, oh)
            self.fft_overlay.move((pw-ow)//2, (ph-oh)//2)

    # ── Acciones ──────────────────────────────────────────
    def _refresh_ports(self):
        self.combo_port.clear()
        ports = serial.tools.list_ports.comports()
        for p in ports: self.combo_port.addItem(f"{p.device}  —  {p.description}", p.device)
        if not ports: self.combo_port.addItem("(no hay puertos disponibles)","")
        self._set_status(f"{len(ports)} puerto(s) detectado(s)","dim")

    def _toggle_conexion(self):
        if self.serial_thread and self.serial_thread.isRunning(): self._desconectar()
        else: self._conectar()

    def _conectar(self):
        port = self.combo_port.currentData()
        if not port: self._set_status("Selecciona un puerto válido","warn"); return
        baud = int(self.combo_baud.currentText())
        self._puerto_activo = port; self._baud_activo = baud
        self.serial_thread = SerialReader(port,baud)
        self.serial_thread.data_received.connect(self._on_data)
        self.serial_thread.error_signal.connect(self._on_serial_error)
        self.serial_thread.start()
        self.btn_conectar.setText("✕  DESCONECTAR"); self.btn_conectar.setStyleSheet(BTN_DANGER)
        self.btn_pausa.setEnabled(True); self.btn_record.setEnabled(True)
        self.lbl_conn.setText("● ONLINE")
        self.lbl_conn.setStyleSheet(f"color:{PALETTE['ok']};letter-spacing:2px;font-weight:bold;")
        self._set_status(f"Conectado  →  {port}  @  {baud} baud","ok")

    def _desconectar(self):
        if self.grabando: self._toggle_grabacion()
        if self.serial_thread: self.serial_thread.stop(); self.serial_thread=None
        self.btn_conectar.setText("⚡  CONECTAR"); self.btn_conectar.setStyleSheet(BTN_OK)
        self.btn_pausa.setEnabled(False); self.btn_record.setEnabled(False)
        self.lbl_conn.setText("● OFFLINE")
        self.lbl_conn.setStyleSheet(f"color:{PALETTE['danger']};letter-spacing:2px;font-weight:bold;")
        self._set_status("Desconectado","dim")

    def _toggle_pausa(self):
        self.pausado = not self.pausado
        self.btn_pausa.setText("▶  REANUDAR" if self.pausado else "⏸  PAUSAR")
        self._set_status("Adquisición pausada" if self.pausado else "Adquisición activa","dim")

    def _toggle_grabacion(self):
        if not self.grabando:
            self.grabando=True; self.csv_rows=[]; self.t_inicio_grab=datetime.now()
            self.sample_index=0; self.btn_record.setText("⏹  DETENER")
            self.lbl_rec_dot.setVisible(True)
            self.btn_export_csv.setEnabled(False); self.btn_export_html.setEnabled(False)
            self._set_status("Grabando datos…","warn")
        else:
            self.grabando=False; self.btn_record.setText("⏺  GRABAR")
            self.lbl_rec_dot.setVisible(False); self.lbl_rec_count.setText("")
            n=len(self.csv_rows)
            self.btn_export_csv.setEnabled(n>0); self.btn_export_html.setEnabled(n>0)
            self._set_status(f"Grabación detenida — {n} muestras · Listo para exportar","ok")

    def _exportar_csv(self):
        if not self.csv_rows: self._set_status("No hay datos","warn"); return
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        path,_=QFileDialog.getSaveFileName(self,"Guardar CSV",f"sensor_sw420_{ts}.csv","CSV Files (*.csv)")
        if not path: return
        try:
            with open(path,"w",newline="",encoding="utf-8") as f:
                w=csv.writer(f); w.writerow(["indice","tiempo_ms","valor_adc"]); w.writerows(self.csv_rows)
            self._set_status(f"CSV exportado → {path}","ok")
        except Exception as e: self._set_status(f"Error: {e}","warn")

    def _exportar_html(self):
        if not self.csv_rows: self._set_status("No hay datos","warn"); return
        ts=datetime.now().strftime("%Y%m%d_%H%M%S")
        path,_=QFileDialog.getSaveFileName(self,"Guardar Reporte HTML",f"vibration_report_{ts}.html","HTML Report (*.html)")
        if not path: return
        self._set_status("Generando reporte HTML…","dim"); QApplication.processEvents()
        gen=HTMLReportGenerator(self.csv_rows,self._puerto_activo,self._baud_activo)
        ok=gen.generar(path)
        if ok:
            kb=os.path.getsize(path)/1024
            self._set_status(f"✅  Reporte generado → {path}  ({kb:.1f} KB)","ok")
        else: self._set_status("❌  Error al generar el reporte","warn")

    # ── Datos entrantes ───────────────────────────────────
    def _on_data(self, valor: float):
        if self.pausado: return
        self.buf_signal.append(valor); self.sample_index+=1; self._fft_counter+=1

        if self.grabando:
            t_ms=int((datetime.now()-self.t_inicio_grab).total_seconds()*1000)
            self.csv_rows.append([self.sample_index,t_ms,valor])
            if self.sample_index%10==0: self.lbl_rec_count.setText(f"{len(self.csv_rows)} pts")

        y = list(self.buf_signal)
        self.curve_signal.setData(y)

        color=(PALETTE['danger'] if valor>3000 else (PALETTE['warn'] if valor>THRESHOLD else PALETTE['ok']))
        self.lbl_valor.setText(str(int(valor)))
        self.lbl_valor.setStyleSheet(f"color:{color};letter-spacing:2px;")

        if self._fft_counter >= FFT_REFRESH_EVERY:
            self._fft_counter = 0
            self._compute_fft_centered(y)

    def _compute_fft_centered(self, signal_data: list):
        arr = np.array(signal_data, dtype=np.float64)
        arr -= arr.mean()

        # ── Energía RMS de la señal (sin DC) ──────────────
        rms = float(np.sqrt(np.mean(arr**2)))

        # ── Si no hay señal real, mostrar overlay y salir ──
        if rms < FFT_MIN_RMS:
            if self._fft_active:
                self._fft_active = False
                self.curve_fft.setData([], [])          # limpiar curva
                self.lbl_dom_freq.setText("Dom. Freq: ——  Hz")
                self.fft_overlay.show()
                self._reposition_overlay()
            return

        # ── Hay señal: ocultar overlay y calcular FFT ─────
        if not self._fft_active:
            self._fft_active = True
            self.fft_overlay.hide()

        N   = len(arr)
        arr *= np.hanning(N)                             # ventana Hanning

        # FFT completa + fftshift → centrada en 0 Hz
        fft_full     = np.fft.fft(arr)
        mag_shifted  = np.fft.fftshift(np.abs(fft_full) / N)
        freq_shifted = np.fft.fftshift(np.fft.fftfreq(N, d=1.0/SAMPLE_RATE_HZ))

        self.curve_fft.setData(freq_shifted, mag_shifted)

        half_fs = SAMPLE_RATE_HZ / 2.0
        self.plot_fft.setXRange(-half_fs, half_fs, padding=0.02)

        # Frecuencia dominante (sólo lado positivo)
        pos_mask = freq_shifted > 0
        if pos_mask.any():
            pos_freqs = freq_shifted[pos_mask]
            pos_mags  = mag_shifted[pos_mask]
            dom_freq  = pos_freqs[np.argmax(pos_mags)]
            self.lbl_dom_freq.setText(f"Dom. Freq:  {dom_freq:.2f}  Hz")

    def _on_serial_error(self, msg):
        self._desconectar(); self._set_status(f"Error serial: {msg}","warn")

    def _set_status(self, msg, level="dim"):
        colors={"ok":PALETTE['ok'],"warn":PALETTE['warn'],"dim":PALETTE['text_dim']}
        color=colors.get(level,PALETTE['text_dim'])
        self.status.setStyleSheet(f"QStatusBar{{background-color:{PALETTE['panel']};border-top:1px solid {PALETTE['border']};color:{color};font-family:'Consolas',monospace;font-size:11px;padding:2px 10px;}}")
        self.status.showMessage(f"[{datetime.now().strftime('%H:%M:%S')}]  {msg}")

    def closeEvent(self, event):
        if self.serial_thread: self.serial_thread.stop()
        super().closeEvent(event)


# ─────────────────────────────────────────────────────────
if __name__ == "__main__":
    app = QApplication(sys.argv); app.setStyle("Fusion")
    pal = QPalette()
    pal.setColor(QPalette.Window,          QColor(7,9,15))
    pal.setColor(QPalette.WindowText,      QColor(201,209,217))
    pal.setColor(QPalette.Base,            QColor(13,17,23))
    pal.setColor(QPalette.AlternateBase,   QColor(22,27,34))
    pal.setColor(QPalette.Text,            QColor(201,209,217))
    pal.setColor(QPalette.ButtonText,      QColor(201,209,217))
    pal.setColor(QPalette.Highlight,       QColor(0,229,255,80))
    pal.setColor(QPalette.HighlightedText, QColor(201,209,217))
    app.setPalette(pal)
    window = MainWindow(); window.show()
    sys.exit(app.exec_())