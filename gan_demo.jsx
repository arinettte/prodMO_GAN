import { useState, useEffect, useRef } from "react";
import {
  ComposedChart, Area, Line, LineChart,
  XAxis, YAxis, Tooltip, ReferenceLine,
  ResponsiveContainer, CartesianGrid, Legend,
} from "recharts";

// ── Математические утилиты ────────────────────────────────────
const sigmoid = x => 1 / (1 + Math.exp(-Math.max(-20, Math.min(20, x))));
const gaussian = (x, mu, sig) =>
  Math.exp(-0.5 * ((x - mu) / sig) ** 2) / (sig * Math.sqrt(2 * Math.PI));
const randn = () => {
  const u = 1 - Math.random(), v = Math.random();
  return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
};

// ── Константы ─────────────────────────────────────────────────
const REAL_MU = 3.0, REAL_SIG = 0.8;
const X_MIN = -4, X_MAX = 8;
const N_PLOT = 180, N_SAMP = 100;
const LR_D = 0.18, LR_G = 0.28;

// ── Один шаг обучения (чистая функция, нет замыканий) ─────────
function step(s) {
  const { gMu, gSig, dW, dB, iter, hist, phase } = s;

  const realX = Array.from({ length: N_SAMP }, () => REAL_MU + REAL_SIG * randn());
  const zs    = Array.from({ length: N_SAMP }, () => randn());
  const fakeX = zs.map(z => gMu + gSig * z);

  // Текущие потери
  const lD = -(
    realX.reduce((a, x) => a + Math.log(sigmoid(dW*x+dB) + 1e-8), 0) / N_SAMP +
    fakeX.reduce((a, x) => a + Math.log(1 - sigmoid(dW*x+dB) + 1e-8), 0) / N_SAMP
  );
  const lG = -fakeX.reduce((a, x) => a + Math.log(sigmoid(dW*x+dB) + 1e-8), 0) / N_SAMP;

  let ns = { ...s, iter: iter + 1 };

  if (phase === "D") {
    // Градиентный подъём по D: max E[log D(x_real)] + E[log(1-D(G(z)))]
    let gW = 0, gB = 0;
    realX.forEach(x => { const d = sigmoid(dW*x+dB); gW += (1-d)*x; gB += (1-d); });
    fakeX.forEach(x => { const d = sigmoid(dW*x+dB); gW -= d*x;     gB -= d;     });
    ns = { ...ns, dW: dW + LR_D*(gW/N_SAMP), dB: dB + LR_D*(gB/N_SAMP), phase: "G" };
  } else {
    // Градиентный подъём по G: max E[log D(G(z))]
    let gM = 0, gS = 0;
    zs.forEach(z => {
      const x = gMu + gSig * z;
      const f = (1 - sigmoid(dW*x+dB)) * dW;
      gM += f; gS += f * z;
    });
    ns = { ...ns,
      gMu:  gMu + LR_G * (gM / N_SAMP),
      gSig: Math.max(0.12, gSig + LR_G * (gS / N_SAMP)),
      phase: "D",
    };
  }

  return {
    ...ns,
    hist: [...hist, { i: iter+1, D: +lD.toFixed(3), G: +lG.toFixed(3) }].slice(-90),
  };
}

// ── Начальное состояние ────────────────────────────────────────
const INIT = { gMu: -1.0, gSig: 1.8, dW: 0.0, dB: 0.0,
               iter: 0, hist: [], phase: "D" };

// ── Компонент ─────────────────────────────────────────────────
export default function GANDemo() {
  const [s, setS] = useState(INIT);
  const [auto, setAuto] = useState(false);
  const iRef = useRef(null);

  useEffect(() => {
    if (auto) iRef.current = setInterval(() => setS(step), 160);
    else clearInterval(iRef.current);
    return () => clearInterval(iRef.current);
  }, [auto]);

  // Данные для графика распределений
  const xs = Array.from({ length: N_PLOT }, (_, i) =>
    X_MIN + (X_MAX - X_MIN) * i / (N_PLOT - 1));
  const distData = xs.map(x => ({
    x: +x.toFixed(2),
    real: +gaussian(x, REAL_MU, REAL_SIG).toFixed(4),
    gen:  +gaussian(x, s.gMu, s.gSig).toFixed(4),
    disc: +sigmoid(s.dW * x + s.dB).toFixed(4),
  }));

  // Метрика перекрытия ∫ min(p_data, p_G) dx
  const overlap = xs.reduce((a, x) =>
    a + Math.min(gaussian(x, REAL_MU, REAL_SIG),
                 gaussian(x, s.gMu, s.gSig)) * (X_MAX - X_MIN) / (N_PLOT - 1), 0);
  const ovPct = Math.min(100, Math.round(overlap * 100));
  const conv = ovPct >= 78, good = ovPct >= 50;
  const ovColor = conv ? "#3fb950" : good ? "#d29922" : "#388bfd";

  // Описание текущей фазы
  const phaseInfo = s.phase === "D"
    ? { label: "Обучаем D", sub: "D старается отличить реальное от фейкового", col: "#3fb950" }
    : { label: "Обучаем G", sub: "G старается обмануть дискриминатор D",       col: "#f47067" };

  const boundary = s.dW !== 0 ? (-s.dB / s.dW).toFixed(2) : "∞";

  return (
    <div style={{
      fontFamily: "'Inter','Segoe UI',sans-serif",
      background: "#0d1117", color: "#c9d1d9",
      minHeight: "100vh", padding: "18px 16px",
      maxWidth: 720, margin: "0 auto",
    }}>

      {/* ── Заголовок ── */}
      <div style={{ textAlign: "center", marginBottom: 14 }}>
        <div style={{ fontSize: 16, fontWeight: 700, color: "#58a6ff" }}>
          Интерактивная демонстрация обучения GAN
        </div>
        <div style={{ fontSize: 11, color: "#6e7681", marginTop: 3 }}>
          1D-пример: G(z) = μ + σ·z &nbsp;|&nbsp; D(x) = σ(wx + b) &nbsp;|&nbsp;
          реальные данные: N({REAL_MU}, {REAL_SIG})
        </div>
      </div>

      {/* ── График распределений ── */}
      <div style={{ background: "#161b22", borderRadius: 10, padding: "12px 6px 6px", marginBottom: 10 }}>
        <div style={{ display: "flex", flexWrap: "wrap", gap: "6px 18px",
                      paddingLeft: 14, marginBottom: 6, fontSize: 11 }}>
          <span style={{ color: "#388bfd" }}>■ p_data(x) — реальные данные</span>
          <span style={{ color: "#f47067" }}>■ p_G(x) — генератор</span>
          <span style={{ color: "#3fb950" }}>╌ D(x) — дискриминатор (ось →)</span>
        </div>
        <ResponsiveContainer width="100%" height={200}>
          <ComposedChart data={distData} margin={{ top: 4, right: 48, bottom: 0, left: 0 }}>
            <XAxis dataKey="x" type="number" domain={[X_MIN, X_MAX]} tickCount={7}
                   tick={{ fontSize: 10, fill: "#6e7681" }} />
            <YAxis yAxisId="dens" domain={[0, 0.65]} width={36}
                   tick={{ fontSize: 10, fill: "#6e7681" }}
                   label={{ value: "Плотность", angle: -90, position: "insideLeft",
                            fontSize: 9, fill: "#6e7681", dy: 32 }} />
            <YAxis yAxisId="disc" orientation="right" domain={[0, 1]} width={36}
                   tick={{ fontSize: 10, fill: "#3fb950" }}
                   label={{ value: "D(x)", angle: 90, position: "insideRight",
                            fontSize: 10, fill: "#3fb950", dy: -12 }} />
            <Tooltip
              contentStyle={{ background: "#161b22", border: "1px solid #30363d", fontSize: 11 }}
              formatter={(v, n) => [v.toFixed(3), n]}
            />
            {/* Целевое положение генератора */}
            <ReferenceLine yAxisId="dens" x={REAL_MU}
              stroke="#388bfd" strokeDasharray="3 3" strokeOpacity={0.35} />
            {/* Граница дискриминатора (D=0.5) */}
            <ReferenceLine yAxisId="disc" y={0.5}
              stroke="#3fb950" strokeDasharray="2 5" strokeOpacity={0.3} />
            <Area yAxisId="dens" type="monotone" dataKey="real"
                  stroke="#388bfd" fill="#388bfd" fillOpacity={0.22}
                  strokeWidth={2} dot={false} name="p_data" />
            <Area yAxisId="dens" type="monotone" dataKey="gen"
                  stroke="#f47067" fill="#f47067" fillOpacity={0.22}
                  strokeWidth={2} dot={false} name="p_G" />
            <Line yAxisId="disc" type="monotone" dataKey="disc"
                  stroke="#3fb950" strokeWidth={1.5} dot={false}
                  strokeDasharray="6 3" name="D(x)" />
          </ComposedChart>
        </ResponsiveContainer>
      </div>

      {/* ── Карточки состояния ── */}
      <div style={{ display: "grid", gridTemplateColumns: "1fr 1fr 1fr", gap: 8, marginBottom: 10 }}>

        {/* Генератор */}
        <div style={{ background: "#161b22", borderRadius: 8, padding: "10px 11px",
                      borderLeft: "3px solid #f47067" }}>
          <div style={{ fontSize: 10, color: "#6e7681", marginBottom: 6,
                        textTransform: "uppercase", letterSpacing: 0.5 }}>Генератор G</div>
          <div style={{ fontSize: 13, marginBottom: 2 }}>
            μ<sub>G</sub> = <b style={{ color: "#f47067" }}>{s.gMu.toFixed(3)}</b>
          </div>
          <div style={{ fontSize: 13, marginBottom: 8 }}>
            σ<sub>G</sub> = <b style={{ color: "#f47067" }}>{s.gSig.toFixed(3)}</b>
          </div>
          <div style={{ fontSize: 10, color: "#6e7681", marginBottom: 5 }}>
            Цель: μ → {REAL_MU}, σ → {REAL_SIG}
          </div>
          {[
            ["μ", s.gMu,  -1.0, REAL_MU,  -1.0],
            ["σ", s.gSig,  0.12, REAL_SIG,  0.12],
          ].map(([lbl, val, lo, hi]) => {
            const pct = Math.max(0, Math.min(100,
              ((val - lo) / (hi - lo)) * 100));
            return (
              <div key={lbl} style={{ display: "flex", alignItems: "center", gap: 5, marginTop: 4 }}>
                <span style={{ fontSize: 9, color: "#6e7681", width: 8 }}>{lbl}</span>
                <div style={{ flex: 1, height: 4, background: "#21262d", borderRadius: 2 }}>
                  <div style={{ width: `${pct}%`, height: "100%", background: "#f47067",
                                borderRadius: 2, transition: "width .25s" }} />
                </div>
                <span style={{ fontSize: 9, color: "#6e7681", width: 24, textAlign: "right" }}>
                  {pct.toFixed(0)}%
                </span>
              </div>
            );
          })}
        </div>

        {/* Дискриминатор */}
        <div style={{ background: "#161b22", borderRadius: 8, padding: "10px 11px",
                      borderLeft: "3px solid #3fb950" }}>
          <div style={{ fontSize: 10, color: "#6e7681", marginBottom: 6,
                        textTransform: "uppercase", letterSpacing: 0.5 }}>Дискриминатор D</div>
          <div style={{ fontSize: 13, marginBottom: 2 }}>
            w = <b style={{ color: "#3fb950" }}>{s.dW.toFixed(3)}</b>
          </div>
          <div style={{ fontSize: 13, marginBottom: 2 }}>
            b = <b style={{ color: "#3fb950" }}>{s.dB.toFixed(3)}</b>
          </div>
          <div style={{ fontSize: 10, color: "#6e7681", marginTop: 8, lineHeight: 1.7 }}>
            D(x) = σ(wx + b)<br />
            Граница: x* = {boundary}
          </div>
          <div style={{ marginTop: 8, fontSize: 11, lineHeight: 1.8 }}>
            D({REAL_MU}) = <b style={{ color: "#388bfd" }}>
              {sigmoid(s.dW*REAL_MU + s.dB).toFixed(2)}</b>
            <br />
            D({s.gMu.toFixed(1)}) = <b style={{ color: "#f47067" }}>
              {sigmoid(s.dW*s.gMu + s.dB).toFixed(2)}</b>
          </div>
        </div>

        {/* Прогресс */}
        <div style={{ background: "#161b22", borderRadius: 8, padding: "10px 11px",
                      borderLeft: `3px solid ${ovColor}` }}>
          <div style={{ fontSize: 10, color: "#6e7681", marginBottom: 6,
                        textTransform: "uppercase", letterSpacing: 0.5 }}>Прогресс</div>
          <div style={{ fontSize: 13 }}>Шаг: <b style={{ color: "#58a6ff" }}>{s.iter}</b></div>
          <div style={{ fontSize: 11, marginTop: 4, marginBottom: 10,
                        color: phaseInfo.col, lineHeight: 1.4 }}>
            <b>{phaseInfo.label}</b><br />
            <span style={{ fontSize: 10, color: "#6e7681" }}>{phaseInfo.sub}</span>
          </div>
          <div style={{ fontSize: 10, color: "#6e7681", marginBottom: 4 }}>
            Перекрытие ∫min(p_data, p_G)dx
          </div>
          <div style={{ height: 8, background: "#21262d", borderRadius: 4, overflow: "hidden" }}>
            <div style={{ width: `${ovPct}%`, height: "100%", background: ovColor,
                          transition: "width .3s, background .3s", borderRadius: 4 }} />
          </div>
          <div style={{ fontSize: 12, marginTop: 4, fontWeight: 600, color: ovColor }}>
            {ovPct}%
            {conv ? " — Равновесие!" : good ? " — Близко!" : ""}
          </div>
          {conv && (
            <div style={{ fontSize: 10, color: "#3fb950", marginTop: 6, lineHeight: 1.5 }}>
              p_G ≈ p_data<br />D(x) ≈ 0.5 везде
            </div>
          )}
        </div>
      </div>

      {/* ── Управление ── */}
      <div style={{ display: "flex", gap: 8, justifyContent: "center", marginBottom: 10 }}>
        <button onClick={() => setS(step)} disabled={auto}
          style={{ padding: "9px 20px", borderRadius: 6, border: "none",
                   background: auto ? "#21262d" : "#388bfd",
                   color: auto ? "#6e7681" : "#fff",
                   fontWeight: 600, cursor: auto ? "default" : "pointer", fontSize: 13 }}>
          ▶ Шаг
        </button>
        <button onClick={() => setAuto(a => !a)}
          style={{ padding: "9px 20px", borderRadius: 6, border: "none",
                   background: auto ? "#f47067" : "#3fb950",
                   color: "#fff", fontWeight: 600, cursor: "pointer", fontSize: 13 }}>
          {auto ? "⏹ Стоп" : "⏩ Авто"}
        </button>
        <button onClick={() => { setS(INIT); setAuto(false); }}
          style={{ padding: "9px 20px", borderRadius: 6,
                   border: "1px solid #30363d", background: "transparent",
                   color: "#8b949e", cursor: "pointer", fontSize: 13 }}>
          ↺ Сброс
        </button>
      </div>

      {/* ── График потерь ── */}
      {s.hist.length > 2 && (
        <div style={{ background: "#161b22", borderRadius: 10, padding: "12px 6px 6px", marginBottom: 10 }}>
          <div style={{ fontSize: 11, color: "#6e7681", paddingLeft: 12, marginBottom: 4 }}>
            Функции потерь по шагам обучения
          </div>
          <ResponsiveContainer width="100%" height={130}>
            <LineChart data={s.hist} margin={{ top: 4, right: 20, bottom: 0, left: 0 }}>
              <XAxis dataKey="i" tick={{ fontSize: 10, fill: "#6e7681" }} />
              <YAxis tick={{ fontSize: 10, fill: "#6e7681" }} domain={["auto", "auto"]} width={36} />
              <CartesianGrid stroke="#21262d" strokeDasharray="3 3" />
              <Tooltip
                contentStyle={{ background: "#161b22", border: "1px solid #30363d", fontSize: 11 }}
              />
              <Legend wrapperStyle={{ fontSize: 10, color: "#8b949e" }} />
              <Line type="monotone" dataKey="D" stroke="#3fb950" strokeWidth={1.5}
                    dot={false} name="Loss D" />
              <Line type="monotone" dataKey="G" stroke="#f47067" strokeWidth={1.5}
                    dot={false} name="Loss G" />
            </LineChart>
          </ResponsiveContainer>
        </div>
      )}

      {/* ── Пояснение формулы ── */}
      <div style={{ background: "#161b22", borderRadius: 8, padding: "10px 14px",
                    fontSize: 11, color: "#6e7681", lineHeight: 1.9 }}>
        <b style={{ color: "#8b949e" }}>Задача минимакса:</b>
        {" "}min<sub>G</sub> max<sub>D</sub> V =
        {" "}𝔼<sub>x∼p_data</sub>[log D(x)] + 𝔼<sub>z∼p_z</sub>[log(1−D(G(z)))]
        <br />
        <b style={{ color: "#8b949e" }}>Равновесие Нэша:</b>
        {" "}p_G = p_data &ensp;⟺&ensp; D*(x) = ½ &ensp;⟺&ensp; Loss G = Loss D ≈ ln 2 ≈ 0.693
        <br />
        <b style={{ color: "#8b949e" }}>В этой демонстрации:</b>
        {" "}G(z) = μ<sub>G</sub> + σ<sub>G</sub>·z (гауссов генератор),
        {" "}D(x) = σ(wx+b) (линейный дискриминатор)
      </div>
    </div>
  );
}
