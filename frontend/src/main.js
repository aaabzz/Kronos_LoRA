import {
  CandlestickSeries,
  ColorType,
  HistogramSeries,
  createChart,
} from "lightweight-charts";
import "./styles.css";

const API_HOST = import.meta.env.VITE_API_HOST || "127.0.0.1:7000";
const POOL_STORAGE_KEY = "kronos-stock-pool";
const STATS_STORAGE_KEY = "kronos-daily-trend-stats";
const SESSION_STATS_STORAGE_KEY = "kronos-session-trend-stats";
const FORECAST_SETTINGS_STORAGE_KEY = "kronos-forecast-settings-by-symbol";
const SYMBOL_PATTERN = /^[0-9A-Z]+\.(SZ|SH|BJ)$/;
const FORECAST_MODES = [
  { key: "greedy", label: "贪心", color: "#fabe48" },
  { key: "median", label: "中位数", color: "#7c9cff" },
  { key: "mean", label: "平均", color: "#d879ff" },
];
const params = new URLSearchParams(window.location.search);
const requestedSymbol = (params.get("symbol") || "").trim().toUpperCase();

function loadStockPool() {
  try {
    const serialized = localStorage.getItem(POOL_STORAGE_KEY);
    if (serialized === null) {
      return [SYMBOL_PATTERN.test(requestedSymbol) ? requestedSymbol : "000001.SZ"];
    }
    const stored = JSON.parse(serialized);
    const valid = stored
      .map((item) => String(item).trim().toUpperCase())
      .filter((item) => SYMBOL_PATTERN.test(item));
    if (!valid.length) return [];
    return [...new Set([
      ...(SYMBOL_PATTERN.test(requestedSymbol) ? [requestedSymbol] : []),
      ...valid,
    ])];
  } catch {
    return [SYMBOL_PATTERN.test(requestedSymbol) ? requestedSymbol : "000001.SZ"];
  }
}

let stockPool = loadStockPool();
let symbol = stockPool.includes(requestedSymbol) ? requestedSymbol : (stockPool[0] || "");
const activatedSymbols = new Set();
const activationPromises = new Map();
let forecastModels = [
  { key: "base", name: "Kronos-base", type: "base" },
  { key: "small", name: "Kronos-small", type: "base" },
];
let forecastOptionsLoaded = false;
const restoredForecastSymbols = new Set();

function loadForecastSettings() {
  try {
    const stored = JSON.parse(localStorage.getItem(FORECAST_SETTINGS_STORAGE_KEY) || "{}") || {};
    // Migrate the former generated default without changing other custom values.
    for (const settings of Object.values(stored)) {
      if (Number(settings?.window) === 480) settings.window = 512;
    }
    return stored;
  } catch {
    return {};
  }
}

const forecastSettingsBySymbol = loadForecastSettings();

function forecastSettings(targetSymbol = symbol) {
  const saved = forecastSettingsBySymbol[targetSymbol] || {};
  const modes = (saved.modes || FORECAST_MODES.map(({ key }) => key))
    .filter((key) => FORECAST_MODES.some((mode) => mode.key === key));
  return {
    window: Number(saved.window) || 512,
    predLen: Number(saved.predLen) || 24,
    sampleCount: Number(saved.sampleCount) || 32,
    modelKey: saved.modelKey || "base",
    modes: modes.length ? modes : FORECAST_MODES.map(({ key }) => key),
  };
}

function saveForecastSettings(targetSymbol, settings) {
  if (!targetSymbol) return;
  forecastSettingsBySymbol[targetSymbol] = settings;
  localStorage.setItem(FORECAST_SETTINGS_STORAGE_KEY, JSON.stringify(forecastSettingsBySymbol));
}

function selectedModesFromControls() {
  return FORECAST_MODES
    .filter(({ key }) => element(`mode-${key}`).checked)
    .map(({ key }) => key);
}

function forgetForecastSettings(targetSymbol) {
  delete forecastSettingsBySymbol[targetSymbol];
  localStorage.setItem(FORECAST_SETTINGS_STORAGE_KEY, JSON.stringify(forecastSettingsBySymbol));
  restoredForecastSymbols.delete(targetSymbol);
}

document.querySelector("#app").innerHTML = `
  <main class="shell">
    <header class="topbar">
      <div class="brand">
        <span class="brand-mark">K</span>
        <div>
          <p class="eyebrow">KRONOS MARKET INTELLIGENCE</p>
          <h1>实时行情监控台</h1>
        </div>
      </div>
      <div class="connection" id="connection" data-state="connecting">
        <span class="status-dot"></span>
        <span id="connection-label">正在连接行情</span>
      </div>
    </header>

    <div class="workspace">
      <aside class="stock-sidebar" aria-label="股票池">
        <div class="pool-heading">
          <div>
            <p class="section-kicker">WATCHLIST</p>
            <h2>股票池</h2>
          </div>
          <span class="pool-count" id="pool-count">0</span>
        </div>

        <div class="stock-list" id="stock-list"></div>

        <form class="add-stock" id="add-stock-form">
          <label for="symbol-input">添加股票代码</label>
          <input
            id="symbol-input"
            placeholder="例如 600519.SH"
            autocomplete="off"
            spellcheck="false"
          />
          <button id="add-symbol" type="submit">
            <span>＋</span> 添加股票
          </button>
          <p class="symbol-feedback" id="symbol-feedback" role="status"></p>
        </form>
      </aside>

      <section class="market-main">
        <section class="instrument-bar">
          <div class="quote-title">
            <div class="ticker-row">
              <strong id="symbol-title">${symbol}</strong>
              <span class="period-pill" id="period">5分钟</span>
              <span class="market-state" id="market-state">等待行情</span>
            </div>
            <div class="last-row">
              <span class="last-price" id="last-price">--</span>
              <span class="change" id="change">--</span>
            </div>
          </div>
          <div class="last-updated">
            <span>最新数据</span>
            <strong id="last-updated">--</strong>
          </div>
        </section>

        <section class="metrics" aria-label="当前K线数据">
          <article><span>开盘</span><strong id="open">--</strong></article>
          <article><span>最高</span><strong id="high">--</strong></article>
          <article><span>最低</span><strong id="low">--</strong></article>
          <article><span>收盘 / 最新</span><strong id="close">--</strong></article>
          <article><span>成交量</span><strong id="volume">--</strong></article>
          <article><span>成交额</span><strong id="amount">--</strong></article>
        </section>

        <section class="stream-card" aria-label="实时OHLCVA数据流">
          <div class="stream-heading">
            <div>
              <p class="section-kicker">LIVE WEBSOCKET PAYLOAD</p>
              <h2>实时 OHLCVA 数据流</h2>
            </div>
            <div class="stream-state">
              <span class="stream-pulse"></span>
              <span id="stream-status">等待实时数据</span>
              <strong id="stream-count">0</strong>
            </div>
          </div>
          <div class="stream-table-wrap" id="stream-table-wrap">
            <table class="stream-table">
              <thead>
                <tr><th>接收时间</th><th>事件</th><th>O</th><th>H</th><th>L</th><th>C</th><th>V</th><th>A</th></tr>
              </thead>
              <tbody id="stream-body">
                <tr class="stream-empty"><td colspan="8">尚未收到 bar_update 或 bar_closed</td></tr>
              </tbody>
            </table>
          </div>
        </section>

        <section class="chart-card">
          <div class="chart-heading">
            <div>
              <p class="section-kicker">REAL-TIME CANDLESTICK</p>
              <h2>价格走势</h2>
            </div>
            <form class="forecast-controls" id="forecast-form">
              <div class="forecast-control-row forecast-control-row-primary">
                <label class="window-control">窗口 <input id="forecast-window" type="number" min="16" max="512" value="512" /></label>
                <label class="prediction-control">预测 <input id="forecast-length" type="number" min="1" max="120" value="24" /></label>
                <label class="sampling-control">采样 <input id="sample-count" type="number" min="1" max="256" value="32" /></label>
                <fieldset class="mode-switches" id="mode-switches">
                  <legend>预测模式</legend>
                  <label><input id="mode-greedy" type="checkbox" checked />贪心</label>
                  <label><input id="mode-median" type="checkbox" checked />中位数</label>
                  <label><input id="mode-mean" type="checkbox" checked />平均</label>
                </fieldset>
              </div>
              <div class="forecast-control-row forecast-control-row-secondary">
                <div class="lora-control-group">
                  <span class="lora-label">模型</span>
                  <select id="forecast-model" aria-label="Kronos / LoRA 模型" disabled><option value="base">Kronos-base</option></select>
                </div>
                <span id="lora-availability">检测模型…</span>
                <button id="forecast-button" type="submit">重新预测</button>
                <span id="forecast-status">等待 Kronos</span>
              </div>
            </form>
            <div class="legend">
              <span><i class="legend-live"></i>实际</span>
              <span><i class="legend-greedy"></i>贪心</span>
              <span><i class="legend-median"></i>中位数</span>
              <span><i class="legend-mean"></i>平均</span>
            </div>
          </div>
          <div class="probability-strip" aria-label="下一根K线涨跌概率">
            <span>下一根K线概率</span>
            <b class="prob-up">上涨 <strong id="probability-up">--</strong></b>
            <b class="prob-down">下跌 <strong id="probability-down">--</strong></b>
            <b class="prob-flat">平盘 <strong id="probability-flat">--</strong></b>
            <small id="probability-samples">等待采样</small>
          </div>
          <div class="charts-grid unified-chart">
            <section class="chart-panel">
              <div class="chart-label"><span class="live-dot"></span>实时 + 贪心 / 中位数 / 平均预测</div>
              <div id="chart" class="chart" role="img" aria-label="实时K线图"></div>
              <div class="empty-state" id="empty-state">
                <div class="pulse-ring"></div>
                <strong>正在等待 QMT 行情</strong>
                <span>请确认 MiniQMT 与 FastAPI 服务已启动</span>
              </div>
            </section>
          </div>
        </section>

        <section class="comparison-card" aria-label="下一根K线预测验证">
          <div class="comparison-heading">
            <div>
              <p class="section-kicker">NEXT BAR VALIDATION</p>
              <h2>下一根 K 线实际值与预测值</h2>
            </div>
            <div class="comparison-result">
              <span id="comparison-time">等待下一根 K 线闭合</span>
            </div>
          </div>
          <div class="comparison-table-wrap">
            <table class="comparison-table">
              <thead><tr><th>指标</th><th>实际值</th><th>贪心预测 / 差值</th><th>中位数预测 / 差值</th><th>平均预测 / 差值</th></tr></thead>
              <tbody id="comparison-body"></tbody>
            </table>
          </div>
          <p class="comparison-note" id="comparison-note">收到预测后，将在对应的下一根真实 K 线定型时自动比较 OHLCVA。</p>
        </section>

        <section class="statistics-card" aria-label="当日趋势预测统计">
          <div class="statistics-heading">
            <div>
              <p class="section-kicker">DAILY TREND ACCURACY</p>
              <h2>当日趋势预测统计</h2>
            </div>
            <span id="statistics-date">--</span>
          </div>
          <div class="mode-statistics" id="mode-statistics"></div>
          <p class="statistics-note">按股票独立统计三种预测模式及下一根 K 线最大概率方向；以交易日期隔日清零，切换股票不清除当日结果。</p>
        </section>

        <section class="session-trend-card" aria-label="上午和下午盘口预测趋势">
          <div class="session-trend-heading">
            <div>
              <p class="section-kicker">SESSION TREND TRACKER</p>
              <h2>上午 / 下午盘口预测趋势</h2>
            </div>
            <span id="session-trend-date">--</span>
          </div>
          <div class="session-trend-grid">
            <article class="session-panel" data-session="morning">
              <div class="session-panel-heading">
                <div><strong>上午盘</strong><span>09:30–11:30 · 24根</span></div>
                <b>三模式对比</b>
              </div>
              <div class="session-mode-table" id="morning-modes"></div>
            </article>
            <article class="session-panel" data-session="afternoon">
              <div class="session-panel-heading">
                <div><strong>下午盘</strong><span>13:00–15:00 · 24根</span></div>
                <b>三模式对比</b>
              </div>
              <div class="session-mode-table" id="afternoon-modes"></div>
            </article>
          </div>
          <p class="session-trend-note">上午严格比较 11:30 预测收盘价与 09:30 实际开盘价；下午严格比较 15:00 预测收盘价与 13:00 实际开盘价。每次实时预测更新方向，只有方向改变时才增加变化统计；盘尾实际走势使用相同开盘价口径验证。</p>
        </section>

        <footer>
          <span>数据源 MiniQMT · WebSocket 实时传输</span>
          <span id="event-kind">尚未收到事件</span>
        </footer>
      </section>
    </div>
  </main>
`;

const element = (id) => document.getElementById(id);
const chartContainer = element("chart");
const emptyState = element("empty-state");

function chartOptions(bottomMargin = 0.24) {
  return {
  autoSize: true,
  layout: {
    background: { type: ColorType.Solid, color: "#0a1511" },
    textColor: "#91a49b",
    attributionLogo: true,
    fontFamily: "Inter, ui-sans-serif, system-ui, sans-serif",
  },
  grid: {
    vertLines: { color: "rgba(132, 160, 146, 0.08)" },
    horzLines: { color: "rgba(132, 160, 146, 0.08)" },
  },
  rightPriceScale: {
    borderColor: "rgba(132, 160, 146, 0.18)",
    scaleMargins: { top: 0.08, bottom: bottomMargin },
  },
  timeScale: {
    borderColor: "rgba(132, 160, 146, 0.18)",
    timeVisible: true,
    secondsVisible: false,
    rightOffset: 4,
  },
  crosshair: {
    vertLine: { color: "rgba(179, 245, 214, 0.3)", labelBackgroundColor: "#24483a" },
    horzLine: { color: "rgba(179, 245, 214, 0.3)", labelBackgroundColor: "#24483a" },
  },
  localization: {
    locale: "zh-CN",
    timeFormatter: (unixTime) => new Intl.DateTimeFormat("zh-CN", {
      timeZone: "UTC",
      month: "2-digit",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit",
      hour12: false,
    }).format(new Date(Number(unixTime) * 1000)),
  },
  };
}

const chart = createChart(chartContainer, chartOptions());
let sharedPriceRange = null;

function sharedAutoscaleInfoProvider() {
  return sharedPriceRange ? { priceRange: sharedPriceRange } : null;
}

const candles = chart.addSeries(CandlestickSeries, {
  upColor: "#ff5f68",
  downColor: "#20c997",
  borderUpColor: "#ff5f68",
  borderDownColor: "#20c997",
  wickUpColor: "#ff7a82",
  wickDownColor: "#46dbaa",
  priceFormat: { type: "price", precision: 2, minMove: 0.01 },
  autoscaleInfoProvider: sharedAutoscaleInfoProvider,
});

const volumes = chart.addSeries(HistogramSeries, {
  priceScaleId: "volume",
  priceFormat: { type: "volume" },
});
volumes.priceScale().applyOptions({ scaleMargins: { top: 0.82, bottom: 0 } });

const forecastSeries = Object.fromEntries(FORECAST_MODES.map((mode, index) => [
  mode.key,
  chart.addSeries(CandlestickSeries, {
    upColor: `${mode.color}${index === 0 ? "70" : "48"}`,
    downColor: `${mode.color}${index === 0 ? "70" : "48"}`,
    borderUpColor: mode.color, borderDownColor: mode.color,
    wickUpColor: mode.color, wickDownColor: mode.color,
    priceLineVisible: false, lastValueVisible: false,
    priceFormat: { type: "price", precision: 2, minMove: 0.01 },
    autoscaleInfoProvider: sharedAutoscaleInfoProvider,
  }),
]));

let socket = null;
let reconnectTimer = null;
let reconnectAttempt = 0;
let connectionGeneration = 0;
let currentTradingDay = null;
let currentDayOpen = null;
const MAX_REALTIME_ROWS = 20;
const HISTORY_BAR_LIMIT = 512;
const symbolStates = new Map();

function loadStatsStore() {
  try {
    return JSON.parse(localStorage.getItem(STATS_STORAGE_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

const statsStore = loadStatsStore();

function loadSessionTrendStore() {
  try {
    return JSON.parse(localStorage.getItem(SESSION_STATS_STORAGE_KEY) || "{}") || {};
  } catch {
    return {};
  }
}

const sessionTrendStore = loadSessionTrendStore();

function persistStatsStore() {
  localStorage.setItem(STATS_STORAGE_KEY, JSON.stringify(statsStore));
}

function persistSessionTrendStore() {
  localStorage.setItem(SESSION_STATS_STORAGE_KEY, JSON.stringify(sessionTrendStore));
}

function chinaDate() {
  return new Intl.DateTimeFormat("en-CA", {
    timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
  }).format(new Date());
}

function eventDate(bar) {
  return String(bar?.timestamp || chinaDate()).slice(0, 10);
}

function ensureStats(targetSymbol, date = chinaDate()) {
  const saved = statsStore[targetSymbol];
  if (!saved || saved.date !== date) {
    statsStore[targetSymbol] = {
      date, calculationVersion: 4, lastCountedTimestamp: null,
      modes: Object.fromEntries(FORECAST_MODES.map(({ key }) => [
        key, { total: 0, correct: 0, lastResult: null },
      ])),
      probability: { total: 0, correct: 0, lastResult: null },
    };
    persistStatsStore();
  } else if (saved.calculationVersion !== 4) {
    for (const { key } of FORECAST_MODES) {
      saved.modes[key] = { ...saved.modes[key], lastResult: null };
    }
    saved.probability = { ...(saved.probability || { total: 0, correct: 0 }), lastResult: null };
    saved.calculationVersion = 4;
    persistStatsStore();
  }
  return statsStore[targetSymbol];
}

function stateFor(targetSymbol) {
  if (!symbolStates.has(targetSymbol)) {
    symbolStates.set(targetSymbol, {
      history: null,
      latestEvent: null,
      barsByTimestamp: new Map(),
      forecastEvent: null,
      forecastStatus: null,
      forecastModesByTimestamp: Object.fromEntries(
        FORECAST_MODES.map(({ key }) => [key, new Map()])
      ),
      probabilityForecastsByTimestamp: new Map(),
      realtimeRows: [],
      receivedEventCount: 0,
      comparison: null,
    });
  }
  return symbolStates.get(targetSymbol);
}

function synchronizePriceScales() {
  if (!symbol || !symbolStates.has(symbol)) {
    sharedPriceRange = null;
  } else {
    const state = symbolStates.get(symbol);
    const bars = [
      ...state.barsByTimestamp.values(),
      ...Object.values(state.forecastEvent?.modes || {}).flat(),
    ];
    const lows = bars.map((bar) => Number(bar.low)).filter(Number.isFinite);
    const highs = bars.map((bar) => Number(bar.high)).filter(Number.isFinite);
    if (lows.length && highs.length) {
      const minimum = Math.min(...lows);
      const maximum = Math.max(...highs);
      const padding = Math.max((maximum - minimum) * 0.03, Math.abs(maximum) * 0.001, 0.01);
      sharedPriceRange = {
        minValue: minimum - padding,
        maxValue: maximum + padding,
      };
    } else {
      sharedPriceRange = null;
    }
  }

  // Reapplying the provider invalidates Lightweight Charts' cached autoscale result.
  candles.applyOptions({ autoscaleInfoProvider: sharedAutoscaleInfoProvider });
  Object.values(forecastSeries).forEach((series) => {
    series.applyOptions({ autoscaleInfoProvider: sharedAutoscaleInfoProvider });
  });
}

const comparisonFields = [
  ["open", "O · 开盘", "price"],
  ["high", "H · 最高", "price"],
  ["low", "L · 最低", "price"],
  ["close", "C · 收盘", "price"],
  ["volume", "V · 成交量", "integer"],
  ["amount", "A · 成交额", "integer"],
];

function resetComparison() {
  element("comparison-time").textContent = "等待下一根 K 线闭合";
  element("comparison-note").textContent = "收到预测后，将在对应的下一根真实 K 线定型时自动比较 OHLCVA；趋势命中结果在当日趋势预测统计中显示。";
  element("comparison-body").innerHTML = comparisonFields.map(([, label]) =>
    `<tr><th>${label}</th><td>--</td><td>-- / --</td><td>-- / --</td><td>-- / --</td></tr>`
  ).join("");
}

function comparisonValue(value, kind) {
  return kind === "price" ? formatPrice(value) : formatNumber(value);
}

function streamTimestamp(bar) {
  return String(bar.timestamp || "--").replace("T", " ").slice(0, 19);
}

function renderRealtimeStream(state = stateFor(symbol)) {
  const body = element("stream-body");
  if (!state.realtimeRows.length) {
    body.innerHTML = '<tr class="stream-empty"><td colspan="8">尚未收到 bar_update 或 bar_closed</td></tr>';
    return;
  }
  body.innerHTML = state.realtimeRows.map(({ type, bar }) => `
    <tr data-event="${type}">
      <td>${streamTimestamp(bar)}</td>
      <td><span class="event-badge">${type}</span></td>
      <td>${formatPrice(bar.open)}</td>
      <td>${formatPrice(bar.high)}</td>
      <td>${formatPrice(bar.low)}</td>
      <td>${formatPrice(bar.close)}</td>
      <td>${formatNumber(bar.volume)}</td>
      <td>${formatNumber(bar.amount)}</td>
    </tr>
  `).join("");
}

function renderStreamState(state = stateFor(symbol)) {
  element("stream-status").textContent = !state.realtimeRows.length
    ? "等待实时数据"
    : state.realtimeRows[0].type === "bar_closed"
      ? "收到已闭合 K 线"
      : "正在接收盘中更新";
  element("stream-count").textContent = String(state.receivedEventCount);
  renderRealtimeStream(state);
}

function appendRealtimeStream(event) {
  const state = stateFor(event.symbol);
  state.realtimeRows.unshift({ type: event.type, bar: event.bar });
  state.realtimeRows = state.realtimeRows.slice(0, MAX_REALTIME_ROWS);
  state.receivedEventCount += 1;
  if (event.symbol !== symbol) return;
  element("stream-status").textContent = event.type === "bar_closed"
    ? "收到已闭合 K 线"
    : "正在接收盘中更新";
  element("stream-count").textContent = String(state.receivedEventCount);
  renderRealtimeStream(state);
}

function maximumProbability(probabilities = {}) {
  const candidates = [
    ["up", Number(probabilities.up)],
    ["down", Number(probabilities.down)],
    ["flat", Number(probabilities.flat)],
  ];
  if (candidates.some(([, value]) => !Number.isFinite(value))) return null;
  // Strictly greater preserves the documented tie priority: up, down, flat.
  const [trend, probability] = candidates.reduce(
    (winner, candidate) => candidate[1] > winner[1] ? candidate : winner
  );
  return { trend, probability, candidates };
}

function forecastReferenceClose(event, state) {
  const basedOnMs = Date.parse(event?.based_on_timestamp);
  const observedBars = [...state.barsByTimestamp.values()]
    .filter((bar) => Number(bar.timestamp_ms) <= basedOnMs)
    .sort((a, b) => Number(b.timestamp_ms) - Number(a.timestamp_ms));
  const referenceClose = Number(observedBars[0]?.close);
  return Number.isFinite(referenceClose) ? referenceClose : null;
}

function currentModeForecastTrend(targetSymbol, modeKey) {
  const settings = forecastSettings(targetSymbol);
  if (!settings.modes.includes(modeKey)) return { state: "disabled", trend: null };
  const state = stateFor(targetSymbol);
  const event = state.forecastEvent;
  if (!event || (event.model_key || "base") !== settings.modelKey) {
    return { state: "waiting", trend: null };
  }
  const firstBar = [...(event.modes?.[modeKey] || [])]
    .sort((a, b) => Number(a.timestamp_ms) - Number(b.timestamp_ms))[0];
  const referenceClose = forecastReferenceClose(event, state);
  if (!firstBar || referenceClose === null) return { state: "waiting", trend: null };
  return { state: "ready", trend: trendBetween(referenceClose, firstBar.close) };
}

function currentProbabilityForecastTrend(targetSymbol) {
  const settings = forecastSettings(targetSymbol);
  const event = stateFor(targetSymbol).forecastEvent;
  if (!event || (event.model_key || "base") !== settings.modelKey) {
    return { state: "waiting", trend: null };
  }
  const maximum = maximumProbability(event.probabilities);
  return maximum
    ? { state: "ready", trend: maximum.trend, probability: maximum.probability }
    : { state: "waiting", trend: null };
}

function recordProbabilityForecast(event) {
  const maximum = maximumProbability(event.probabilities);
  if (!maximum) return;
  const { trend: predictedTrend, probability, candidates } = maximum;
  const forecastBars = Object.values(event.modes || {})
    .flat()
    .filter((bar) => Number.isFinite(Number(bar.timestamp_ms)))
    .sort((a, b) => Number(a.timestamp_ms) - Number(b.timestamp_ms));
  const targetBar = forecastBars[0];
  if (!targetBar) return;

  const state = stateFor(event.symbol);
  const referenceClose = forecastReferenceClose(event, state);
  if (referenceClose === null) return;

  state.probabilityForecastsByTimestamp.set(Number(targetBar.timestamp_ms), {
    predictedTrend,
    probability,
    probabilities: Object.fromEntries(candidates),
    referenceClose,
    basedOnTimestamp: event.based_on_timestamp,
  });
  while (state.probabilityForecastsByTimestamp.size > HISTORY_BAR_LIMIT) {
    state.probabilityForecastsByTimestamp.delete(
      state.probabilityForecastsByTimestamp.keys().next().value
    );
  }
}

function trendLabel(trend) {
  return { up: "上涨", down: "下跌", flat: "平盘" }[trend] || "未知";
}

function emptySessionTrend() {
  return {
    currentTrend: null,
    samples: 0,
    changes: 0,
    upChanges: 0,
    downChanges: 0,
    flatChanges: 0,
    latestPrediction: null,
    result: null,
  };
}

function emptySessionModes() {
  return Object.fromEntries(
    FORECAST_MODES.map(({ key }) => [key, emptySessionTrend()])
  );
}

function ensureSessionTrends(targetSymbol, date = chinaDate()) {
  const saved = sessionTrendStore[targetSymbol];
  if (!saved || saved.date !== date || saved.calculationVersion !== 4) {
    sessionTrendStore[targetSymbol] = {
      date,
      calculationVersion: 4,
      morning: emptySessionModes(),
      afternoon: emptySessionModes(),
    };
    persistSessionTrendStore();
  }
  return sessionTrendStore[targetSymbol];
}

function clearSessionMode(targetSymbol, modeKey) {
  const sessions = ensureSessionTrends(targetSymbol);
  sessions.morning[modeKey] = emptySessionTrend();
  sessions.afternoon[modeKey] = emptySessionTrend();
  persistSessionTrendStore();
}

function timestampParts(value) {
  const text = String(value || "");
  return { date: text.slice(0, 10), clock: text.slice(11, 16) };
}

function sessionForTimestamp(value, { closingOnly = false } = {}) {
  const { date, clock } = timestampParts(value);
  if (closingOnly) {
    if (["11:25", "11:30"].includes(clock)) {
      return { key: "morning", date, start: "09:30", end: "11:30" };
    }
    if (["14:55", "15:00"].includes(clock)) {
      return { key: "afternoon", date, start: "13:00", end: "15:00" };
    }
    return null;
  }
  if (clock >= "09:30" && clock < "11:30") {
    return { key: "morning", date, start: "09:30", end: "11:30" };
  }
  if (clock >= "13:00" && clock < "15:00") {
    return { key: "afternoon", date, start: "13:00", end: "15:00" };
  }
  return null;
}

function trendBetween(start, end) {
  const difference = Number(end) - Number(start);
  return difference > 0.005 ? "up" : difference < -0.005 ? "down" : "flat";
}

function sessionOpeningPrice(state, session) {
  const bars = [
    ...(state.history?.bars || []),
    ...state.realtimeRows.map((item) => item.bar),
    ...(state.latestEvent?.bar ? [state.latestEvent.bar] : []),
  ];
  const openingBar = bars
    .filter((bar) => {
      const parts = timestampParts(bar.timestamp);
      return parts.date === session.date
        && parts.clock >= session.start
        && parts.clock < session.end;
    })
    .sort((a, b) => Date.parse(a.timestamp) - Date.parse(b.timestamp))[0];
  if (!openingBar) return null;
  const openingClock = timestampParts(openingBar.timestamp).clock;
  const endLabel = openingClock !== session.start;
  return {
    price: Number(openingBar.open),
    timestamp: openingBar.timestamp,
    closingClock: endLabel
      ? session.end
      : session.key === "morning" ? "11:25" : "14:55",
  };
}

function recordSessionForecast(event) {
  const session = sessionForTimestamp(event.based_on_timestamp);
  if (!session) return;
  const state = stateFor(event.symbol);
  const opening = sessionOpeningPrice(state, session);
  if (!opening || !Number.isFinite(opening.price)) return;
  const sessionModes = ensureSessionTrends(event.symbol, session.date)[session.key];
  const enabledModes = new Set(forecastSettings(event.symbol).modes);
  for (const mode of FORECAST_MODES) {
    if (!enabledModes.has(mode.key)) continue;
    const targetBar = (event.modes?.[mode.key] || []).find((bar) => {
      const parts = timestampParts(bar.timestamp);
      return parts.date === session.date && parts.clock === opening.closingClock;
    });
    if (!targetBar) continue;
    const nextTrend = trendBetween(opening.price, targetBar.close);
    const statistics = sessionModes[mode.key];
    statistics.samples += 1;
    if (statistics.currentTrend && statistics.currentTrend !== nextTrend) {
      statistics.changes += 1;
      statistics[`${nextTrend}Changes`] += 1;
    }
    statistics.currentTrend = nextTrend;
    statistics.latestPrediction = {
      trend: nextTrend, openingPrice: opening.price,
      openingTimestamp: opening.timestamp, closingClock: opening.closingClock,
      projectedClose: Number(targetBar.close),
      basedOnTimestamp: event.based_on_timestamp, targetTimestamp: targetBar.timestamp,
    };
  }
  persistSessionTrendStore();
  const closedSessionBar = [
    ...(state.history?.bars || []),
    ...state.realtimeRows
      .filter((item) => item.type === "bar_closed")
      .map((item) => item.bar),
    ...(state.latestEvent?.type === "bar_closed" ? [state.latestEvent.bar] : []),
  ].find((bar) => {
    const parts = timestampParts(bar.timestamp);
    return parts.date === session.date && parts.clock === opening.closingClock;
  });
  if (closedSessionBar) compareSessionClose(closedSessionBar);
  if (event.symbol === symbol) renderSessionTrends();
}

function compareSessionClose(actual) {
  const session = sessionForTimestamp(actual.timestamp, { closingOnly: true });
  if (!session) return;
  const sessionModes = ensureSessionTrends(actual.symbol, session.date)[session.key];
  const enabledModes = new Set(forecastSettings(actual.symbol).modes);
  for (const mode of FORECAST_MODES) {
    if (!enabledModes.has(mode.key)) continue;
    const statistics = sessionModes[mode.key];
    const prediction = statistics.latestPrediction;
    if (!prediction || timestampParts(actual.timestamp).clock !== prediction.closingClock) continue;
    const actualTrend = trendBetween(prediction.openingPrice, actual.close);
    statistics.result = {
      correct: prediction.trend === actualTrend, predictedTrend: prediction.trend,
      actualTrend, openingPrice: prediction.openingPrice,
      projectedClose: prediction.projectedClose, actualClose: Number(actual.close),
      timestamp: actual.timestamp,
    };
  }
  persistSessionTrendStore();
}

function renderSessionTrends() {
  element("session-trend-date").textContent = symbol
    ? `${symbol} · ${chinaDate()}`
    : `-- · ${chinaDate()}`;
  const stored = symbol ? ensureSessionTrends(symbol) : null;
  const enabledModes = new Set(symbol ? forecastSettings(symbol).modes : []);
  for (const key of ["morning", "afternoon"]) {
    const modes = stored?.[key] || emptySessionModes();
    element(`${key}-modes`).innerHTML = FORECAST_MODES.map((mode) => {
      if (!enabledModes.has(mode.key)) {
        return `<div class="session-mode-row" data-enabled="false">
          <strong style="--mode-color:${mode.color}">${mode.label}</strong>
          <span>未启用</span>
          <span>未启用</span>
          <span>未启用</span>
          <b data-result="disabled">未启用</b>
        </div>`;
      }
      const statistics = modes[mode.key];
      const prediction = statistics.latestPrediction;
      const result = statistics.result;
      return `<div class="session-mode-row">
        <strong style="--mode-color:${mode.color}">${mode.label}</strong>
        <span>当前 ${statistics.currentTrend ? trendLabel(statistics.currentTrend) : "等待"}</span>
        <span>更新 ${statistics.samples} / 变化 ${statistics.changes}</span>
        <span>${prediction ? `预测收盘 ${formatPrice(prediction.projectedClose)}` : "等待预测"}</span>
        <b data-result="${result ? (result.correct ? "correct" : "wrong") : "waiting"}">${result ? `${result.correct ? "✓" : "✕"} 预测${trendLabel(result.predictedTrend)} / 实际${trendLabel(result.actualTrend)}` : `等待 ${key === "morning" ? "11:30" : "15:00"}`}</b>
      </div>`;
    }).join("");
  }
}

function compareClosedBar(actual) {
  const targetSymbol = actual.symbol;
  const state = stateFor(targetSymbol);
  const actualTimestamp = Number(actual.timestamp_ms);
  const previousBar = [...state.barsByTimestamp.values()]
    .filter((bar) => Number(bar.timestamp_ms) < actualTimestamp)
    .sort((a, b) => Number(b.timestamp_ms) - Number(a.timestamp_ms))[0];
  const previousClose = Number(previousBar?.close);
  if (!Number.isFinite(previousClose)) return null;
  const actualTrend = trendBetween(previousClose, actual.close);
  const probabilityPrediction = state.probabilityForecastsByTimestamp.get(
    Number(actual.timestamp_ms)
  );
  const probabilityActualTrend = probabilityPrediction
    ? actualTrend
    : null;
  const probabilityResult = probabilityPrediction ? {
    ...probabilityPrediction,
    actualTrend: probabilityActualTrend,
    correct: probabilityPrediction.predictedTrend === probabilityActualTrend,
  } : null;
  const predictions = Object.fromEntries(FORECAST_MODES.map(({ key }) => [
    key, state.forecastModesByTimestamp[key].get(Number(actual.timestamp_ms)),
  ]));
  if (!Object.values(predictions).some(Boolean)) return null;
  const results = Object.fromEntries(FORECAST_MODES.map(({ key }) => {
    const predicted = predictions[key];
    const predictedTrend = predicted ? trendBetween(previousClose, predicted.close) : null;
    return [key, { predicted, predictedTrend, correct: predictedTrend === actualTrend }];
  }));
  const comparison = { actual, actualTrend, results, probabilityResult };
  state.comparison = comparison;

  const stats = ensureStats(targetSymbol, eventDate(actual));
  if (stats.lastCountedTimestamp !== Number(actual.timestamp_ms)) {
    for (const { key } of FORECAST_MODES) {
      if (!results[key].predicted) continue;
      stats.modes[key].total += 1;
      if (results[key].correct) stats.modes[key].correct += 1;
      stats.modes[key].lastResult = {
        predictedTrend: results[key].predictedTrend,
        actualTrend,
        correct: results[key].correct,
        timestamp: actual.timestamp,
      };
    }
    if (probabilityResult) {
      stats.probability.total += 1;
      if (probabilityResult.correct) stats.probability.correct += 1;
      stats.probability.lastResult = {
        predictedTrend: probabilityResult.predictedTrend,
        actualTrend: probabilityResult.actualTrend,
        correct: probabilityResult.correct,
        probability: probabilityResult.probability,
        timestamp: actual.timestamp,
      };
    }
    stats.lastCountedTimestamp = Number(actual.timestamp_ms);
    persistStatsStore();
  }
  state.probabilityForecastsByTimestamp.delete(Number(actual.timestamp_ms));
  return comparison;
}

function renderComparison(comparison = stateFor(symbol).comparison) {
  if (!comparison) {
    resetComparison();
    return;
  }
  const { actual, actualTrend, results, probabilityResult } = comparison;
  element("comparison-time").textContent = `${actual.symbol || symbol} · ${String(actual.timestamp).replace("T", " ").slice(0, 19)}`;
  element("comparison-body").innerHTML = comparisonFields.map(([field, label, kind]) => {
    const actualValue = Number(actual[field]);
    const modeCells = FORECAST_MODES.map(({ key }) => {
      const predictedValue = Number(results[key].predicted?.[field]);
      if (!Number.isFinite(predictedValue)) return "<td>--</td>";
      const difference = actualValue - predictedValue;
      const direction = difference > 0 ? "positive" : difference < 0 ? "negative" : "flat";
      const prefix = difference > 0 ? "+" : "";
      return `<td data-direction="${direction}">${comparisonValue(predictedValue, kind)} / ${prefix}${comparisonValue(difference, kind)}</td>`;
    }).join("");
    return `<tr>
      <th>${label}</th>
      <td>${comparisonValue(actualValue, kind)}</td>
      ${modeCells}
    </tr>`;
  }).join("");
  element("comparison-note").textContent = "每格显示预测值 /（实际值 − 预测值）；方向统一按本根实际或预测收盘价减去上一根实际收盘价计算，命中结果见当日趋势预测统计。";
}

function renderStatistics() {
  if (!symbol) {
    element("statistics-date").textContent = `-- · ${chinaDate()}`;
    element("mode-statistics").replaceChildren();
    return;
  }
  const stats = ensureStats(symbol);
  element("statistics-date").textContent = `${symbol} · ${stats.date}`;
  const summary = (lastResult) => lastResult
    ? `预测${trendLabel(lastResult.predictedTrend)}/实际${trendLabel(lastResult.actualTrend)}`
    : "等待验证";
  const currentSummary = (current) => current.state === "disabled"
    ? "未启用"
    : current.trend ? `当前预测${trendLabel(current.trend)}` : "等待当前预测";
  const valuesMarkup = (values) => {
    const wrong = values.total - values.correct;
    const accuracy = values.total
      ? `${((values.correct / values.total) * 100).toFixed(1)}%` : "--";
    return `<div class="stat-values">
      <span>已验证<strong>${values.total}</strong></span>
      <span>正确<strong>${values.correct}</strong></span>
      <span>错误<strong>${wrong}</strong></span>
      <span>正确率<strong>${accuracy}</strong></span>
    </div>`;
  };
  const modeCards = FORECAST_MODES.map((mode) => {
    const values = stats.modes[mode.key];
    const current = currentModeForecastTrend(symbol, mode.key);
    return `<article>
      <div class="stat-mode-heading">
        <h3 style="--mode-color:${mode.color}">${mode.label}</h3>
        <b data-trend="${current.trend || current.state}">${currentSummary(current)}</b>
      </div>
      <div class="stat-last-result" data-result="${values.lastResult ? (values.lastResult.correct ? "correct" : "wrong") : "waiting"}">${summary(values.lastResult)}</div>
      ${valuesMarkup(values)}
    </article>`;
  }).join("");
  const probability = stats.probability;
  const currentProbability = currentProbabilityForecastTrend(symbol);
  element("mode-statistics").innerHTML = `${modeCards}<article>
    <div class="stat-mode-heading">
      <h3 style="--mode-color:#55c996">最大概率</h3>
      <b data-trend="${currentProbability.trend || currentProbability.state}">${currentSummary(currentProbability)}</b>
    </div>
    <div class="stat-last-result" data-result="${probability.lastResult ? (probability.lastResult.correct ? "correct" : "wrong") : "waiting"}">${summary(probability.lastResult)}</div>
    ${valuesMarkup(probability)}
  </article>`;
}

function persistStockPool() {
  localStorage.setItem(POOL_STORAGE_KEY, JSON.stringify(stockPool));
}

function renderStockPool() {
  const list = element("stock-list");
  list.replaceChildren();
  for (const code of stockPool) {
    const forecastState = stateFor(code).forecastStatus?.state || "waiting";
    const visualState = forecastState === "running"
      ? "running"
      : forecastState === "completed"
        ? "completed"
        : forecastState === "error"
          ? "error"
          : "waiting";
    const row = document.createElement("div");
    row.className = "stock-row";
    row.dataset.active = String(code === symbol);
    row.dataset.forecastState = visualState;

    const selectButton = document.createElement("button");
    selectButton.type = "button";
    selectButton.className = "stock-item";
    selectButton.dataset.symbol = code;
    selectButton.setAttribute("aria-pressed", String(code === symbol));
    selectButton.title = {
      waiting: "等待预测", running: "正在预测",
      completed: "预测完成", error: "预测失败",
    }[visualState];
    selectButton.innerHTML = `
      <span class="exchange">${code.split(".").at(-1)}</span>
      <span class="stock-code">${code}</span>
      <span class="stock-arrow">›</span>
    `;
    selectButton.addEventListener("click", () => switchSymbol(code));

    const removeButton = document.createElement("button");
    removeButton.type = "button";
    removeButton.className = "remove-stock";
    removeButton.setAttribute("aria-label", `从股票池移除 ${code}`);
    removeButton.title = `移除 ${code} 并停止监听`;
    removeButton.textContent = "×";
    removeButton.addEventListener("click", () => removeStock(code));

    row.append(selectButton, removeButton);
    list.append(row);
  }
  element("pool-count").textContent = String(stockPool.length);
}

function unixTime(bar) {
  const match = String(bar.timestamp).match(
    /^(\d{4})-(\d{2})-(\d{2})T(\d{2}):(\d{2}):(\d{2})/
  );
  if (match) {
    const [, year, month, day, hour, minute, second] = match;
    return Date.UTC(
      Number(year), Number(month) - 1, Number(day),
      Number(hour), Number(minute), Number(second),
    ) / 1000;
  }
  return Math.floor(Number(bar.timestamp_ms) / 1000) + 8 * 60 * 60;
}

function candlePoint(bar) {
  return {
    time: unixTime(bar),
    open: Number(bar.open), high: Number(bar.high),
    low: Number(bar.low), close: Number(bar.close),
  };
}

function volumePoint(bar) {
  return {
    time: unixTime(bar),
    value: Number(bar.volume),
    color: Number(bar.close) >= Number(bar.open)
      ? "rgba(255, 95, 104, 0.34)"
      : "rgba(32, 201, 151, 0.34)",
  };
}

function setConnection(state, label) {
  element("connection").dataset.state = state;
  element("connection-label").textContent = label;
}

function setFeedback(message, state = "idle") {
  const feedback = element("symbol-feedback");
  feedback.textContent = message;
  feedback.dataset.state = state;
}

function formatNumber(value) {
  return new Intl.NumberFormat("zh-CN", { maximumFractionDigits: 0 }).format(value);
}

function formatPrice(value) {
  return Number(value).toFixed(2);
}

function tradingDay(bar) {
  const timestamp = Number(bar?.timestamp_ms);
  if (Number.isFinite(timestamp)) {
    return new Intl.DateTimeFormat("en-CA", {
      timeZone: "Asia/Shanghai", year: "numeric", month: "2-digit", day: "2-digit",
    }).format(new Date(timestamp));
  }
  return String(bar?.timestamp || "").slice(0, 10);
}

function setDayOpenFromBars(bars, latest) {
  const day = tradingDay(latest);
  const firstBar = bars.find((bar) => tradingDay(bar) === day);
  currentTradingDay = day;
  currentDayOpen = Number(firstBar?.open ?? latest.open);
}

function updateQuote(bar, type) {
  const currentClose = Number(bar.close);
  const day = tradingDay(bar);
  if (day !== currentTradingDay || !Number.isFinite(currentDayOpen)) {
    currentTradingDay = day;
    currentDayOpen = Number(bar.open);
  }
  const base = currentDayOpen;
  const delta = currentClose - base;
  const percent = base ? (delta / base) * 100 : 0;
  const direction = delta > 0 ? "up" : delta < 0 ? "down" : "flat";

  element("symbol-title").textContent = bar.symbol || symbol;
  element("period").textContent = bar.period === "5m" ? "5分钟" : bar.period;
  element("last-price").textContent = formatPrice(currentClose);
  element("last-price").dataset.direction = direction;
  element("change").textContent = `${delta >= 0 ? "+" : ""}${delta.toFixed(2)}  ${percent >= 0 ? "+" : ""}${percent.toFixed(2)}%`;
  element("change").dataset.direction = direction;
  element("open").textContent = formatPrice(bar.open);
  element("high").textContent = formatPrice(bar.high);
  element("low").textContent = formatPrice(bar.low);
  element("close").textContent = formatPrice(bar.close);
  element("volume").textContent = formatNumber(bar.volume);
  element("amount").textContent = `¥ ${formatNumber(bar.amount)}`;
  element("last-updated").textContent = new Date(bar.timestamp).toLocaleString("zh-CN", {
    timeZone: "Asia/Shanghai", hour12: false,
  });
  element("market-state").textContent = type === "bar_closed" ? "K线已闭合" : "盘中更新";
  element("event-kind").textContent = `${type} · ${bar.timestamp}`;
  emptyState.hidden = true;
}

function resetMarketView() {
  sharedPriceRange = null;
  candles.applyOptions({ autoscaleInfoProvider: sharedAutoscaleInfoProvider });
  candles.setData([]);
  volumes.setData([]);
  Object.values(forecastSeries).forEach((series) => series.setData([]));
  currentTradingDay = null;
  currentDayOpen = null;
  for (const id of ["last-price", "change", "open", "high", "low", "close", "volume", "amount", "last-updated"]) {
    element(id).textContent = "--";
  }
  element("symbol-title").textContent = symbol;
  element("market-state").textContent = "等待行情";
  element("event-kind").textContent = "正在切换股票";
  emptyState.hidden = false;
  element("forecast-status").textContent = "等待 Kronos";
  applyForecastSettingsToControls();
  for (const id of ["probability-up", "probability-down", "probability-flat"]) {
    element(id).textContent = "--";
  }
  element("probability-samples").textContent = "等待采样";
  renderStreamState();
  renderComparison();
  renderStatistics();
  renderSessionTrends();
}

function applyHistory(event) {
  const bars = [...(event.bars || [])].sort((a, b) => a.timestamp_ms - b.timestamp_ms);
  if (!bars.length) return;
  synchronizePriceScales();
  candles.setData(bars.map(candlePoint));
  volumes.setData(bars.map(volumePoint));
  setDayOpenFromBars(bars, bars.at(-1));
  updateQuote(bars.at(-1), "history_snapshot");
  chart.timeScale().fitContent();
}

function applyRealtime(event) {
  synchronizePriceScales();
  candles.update(candlePoint(event.bar));
  volumes.update(volumePoint(event.bar));
  updateQuote(event.bar, event.type);
  chart.timeScale().scrollToRealTime();
}

function applyCachedMarketState(state) {
  const bars = [...state.barsByTimestamp.values()]
    .sort((a, b) => Number(a.timestamp_ms) - Number(b.timestamp_ms));
  if (!bars.length) return;
  synchronizePriceScales();
  candles.setData(bars.map(candlePoint));
  volumes.setData(bars.map(volumePoint));

  const latest = state.latestEvent?.bar || bars.at(-1);
  setDayOpenFromBars(bars, latest);
  updateQuote(latest, state.latestEvent?.type || "history_snapshot");
  chart.timeScale().fitContent();
}

function applyForecast(event) {
  const modes = event.modes || Object.fromEntries(
    FORECAST_MODES.map(({ key }) => [key, event.bars || []])
  );
  synchronizePriceScales();
  const enabledModes = new Set(forecastSettings(event.symbol).modes);
  for (const mode of FORECAST_MODES) {
    const bars = enabledModes.has(mode.key)
      ? [...(modes[mode.key] || [])].sort((a, b) => a.timestamp_ms - b.timestamp_ms)
      : [];
    forecastSeries[mode.key].setData(bars.map(candlePoint));
  }
  const probabilities = event.probabilities || {};
  element("probability-up").textContent = Number.isFinite(probabilities.up) ? `${(probabilities.up * 100).toFixed(1)}%` : "--";
  element("probability-down").textContent = Number.isFinite(probabilities.down) ? `${(probabilities.down * 100).toFixed(1)}%` : "--";
  element("probability-flat").textContent = Number.isFinite(probabilities.flat) ? `${(probabilities.flat * 100).toFixed(1)}%` : "--";
  element("probability-samples").textContent = probabilities.sample_count ? `${probabilities.sample_count} 条采样路径` : "等待采样";
  const modelKey = event.model_key || (event.use_lora ? "lora:legacy" : "base");
  const selectedModes = event.selected_modes || Object.keys(modes);
  const savedSettings = forecastSettings(event.symbol);
  // A completed older task must not overwrite a newer model choice made by the user.
  if (modelKey === savedSettings.modelKey) {
    element("forecast-window").value = String(savedSettings.window);
    element("forecast-length").value = String(savedSettings.predLen);
    element("sample-count").value = String(savedSettings.sampleCount);
    element("forecast-model").value = savedSettings.modelKey;
    FORECAST_MODES.forEach(({ key }) => {
      element(`mode-${key}`).checked = savedSettings.modes.includes(key);
    });
  }
  element("forecast-status").textContent = `${modelKey === "base" ? "" : `${modelName(modelKey)} · `}${selectedModes.length}种模式已生成 ${event.pred_len} 根 · 基于 ${String(event.based_on_timestamp).slice(5, 16).replace("T", " ")}`;
  chart.timeScale().fitContent();
}

function rememberEvent(event) {
  if (!event.symbol) return;
  const state = stateFor(event.symbol);
  if (event.type === "history_snapshot") {
    state.history = event;
    state.barsByTimestamp = new Map(
      (event.bars || []).map((bar) => [Number(bar.timestamp_ms), bar])
    );
    restoreSavedForecast(event.symbol);
  } else if (event.type === "bar_update" || event.type === "bar_closed") {
    state.latestEvent = event;
    state.barsByTimestamp.set(Number(event.bar.timestamp_ms), event.bar);
    // Retain 512 closed history bars plus at most one currently forming bar.
    while (state.barsByTimestamp.size > HISTORY_BAR_LIMIT + 1) {
      state.barsByTimestamp.delete(state.barsByTimestamp.keys().next().value);
    }
    appendRealtimeStream(event);
    if (event.type === "bar_closed") {
      compareClosedBar(event.bar);
      compareSessionClose(event.bar);
    }
  } else if (event.type === "forecast") {
    state.forecastEvent = event;
    recordProbabilityForecast(event);
    const modes = event.modes || Object.fromEntries(
      FORECAST_MODES.map(({ key }) => [key, event.bars || []])
    );
    state.forecastModesByTimestamp = Object.fromEntries(
      FORECAST_MODES.map(({ key }) => [key, new Map(
        (modes[key] || []).map((bar) => [Number(bar.timestamp_ms), bar])
      )])
    );
    recordSessionForecast(event);
    state.forecastStatus = { state: "completed" };
  } else if (event.type === "forecast_status") {
    state.forecastStatus = event;
  }
  if (event.type === "forecast" || event.type === "forecast_status") renderStockPool();
}

function renderSymbolView() {
  resetMarketView();
  if (!symbol) return;
  const state = stateFor(symbol);
  applyCachedMarketState(state);
  if (state.forecastEvent) applyForecast(state.forecastEvent);
  if (state.forecastStatus?.state !== "running" || !state.forecastEvent) {
    if (state.forecastStatus) applyForecastStatus(state.forecastStatus);
  }
  renderStreamState(state);
  renderComparison(state.comparison);
  renderStatistics();
  renderSessionTrends();
}

function applyForecastStatus(event) {
  const status = element("forecast-status");
  if (event.state === "running") {
    status.textContent = `${modelName(event.model_key || "base")} 推理中（${event.window} → ${event.pred_len}）`;
    status.dataset.state = "running";
  } else if (event.state === "error") {
    status.textContent = `预测失败：${event.message}`;
    status.dataset.state = "error";
  } else if (event.state === "insufficient_data") {
    status.textContent = `数据不足：${event.message}`;
    status.dataset.state = "error";
  }
}

async function requestForecast() {
  const windowSize = Number(element("forecast-window").value);
  const predLen = Number(element("forecast-length").value);
  const modelKey = element("forecast-model").value || "base";
  const sampleCount = Number(element("sample-count").value);
  const modes = selectedModesFromControls();
  if (!modes.length) throw new Error("请至少选择一种预测模式");
  saveForecastSettings(symbol, { window: windowSize, predLen, modelKey, modes, sampleCount });
  const response = await fetch(
    `http://${API_HOST}/api/forecast/${encodeURIComponent(symbol)}?window=${windowSize}&pred_len=${predLen}&model_key=${encodeURIComponent(modelKey)}&modes=${encodeURIComponent(modes.join(","))}&sample_count=${sampleCount}`,
    { method: "POST" },
  );
  let result = {};
  try { result = await response.json(); } catch { /* use generic error */ }
  if (!response.ok) throw new Error(result.detail || "无法启动 Kronos 预测");
  element("forecast-status").textContent = `${modelName(modelKey)}预测任务已排队`;
}

async function loadForecastOptions() {
  const select = element("forecast-model");
  const availability = element("lora-availability");
  select.disabled = true;
  try {
    const response = await fetch(`http://${API_HOST}/api/forecast/options`);
    if (!response.ok) throw new Error("无法读取LoRA配置");
    const options = await response.json();
    if (!Array.isArray(options.models)) {
      throw new Error("后端版本过旧，请重启 FastAPI 服务");
    }
    const loraModels = (options.models || []).filter((model) => model.type === "lora");
    const baseModels = (options.models || []).filter((model) => model.type === "base");
    forecastModels = [...baseModels, ...loraModels];
    select.replaceChildren(...forecastModels.map((model) => {
      const option = document.createElement("option");
      option.value = model.key;
      option.textContent = model.type === "lora" && model.directory
        ? `LoRA · ${model.name}（${model.directory}）`
        : model.name;
      option.title = model.directory || model.name;
      return option;
    }));
    select.disabled = false;
    forecastOptionsLoaded = true;
    availability.textContent = loraModels.length
      ? `${loraModels.length} 个 LoRA 可选`
      : "未找到 LoRA 文件";
    availability.dataset.state = loraModels.length ? "available" : "unavailable";
    select.parentElement.title = forecastModels.map((model) => model.name).join("、");
    applyForecastSettingsToControls();
    for (const [targetSymbol, state] of symbolStates) {
      if (state.history) restoreSavedForecast(targetSymbol);
    }
  } catch (error) {
    availability.textContent = "LoRA检测失败";
    availability.dataset.state = "unavailable";
    select.parentElement.title = error.message;
  }
}

function modelName(modelKey) {
  return forecastModels.find((model) => model.key === modelKey)?.name || modelKey || "Kronos";
}

function applyForecastSettingsToControls() {
  const settings = forecastSettings();
  element("forecast-window").value = String(settings.window);
  element("forecast-length").value = String(settings.predLen);
  element("sample-count").value = String(settings.sampleCount);
  FORECAST_MODES.forEach(({ key }) => {
    element(`mode-${key}`).checked = settings.modes.includes(key);
    if (symbol && !settings.modes.includes(key)) clearSessionMode(symbol, key);
  });
  const select = element("forecast-model");
  if (!forecastOptionsLoaded) {
    // Render the locally remembered choice immediately. The backend catalog
    // will replace this temporary option and validate it before subscribing.
    const remembered = document.createElement("option");
    const knownModel = forecastModels.find((model) => model.key === settings.modelKey);
    remembered.value = settings.modelKey;
    remembered.textContent = settings.modelKey.startsWith("lora:")
      ? `LoRA · ${settings.modelKey.replace(/^lora:/, "")}`
      : knownModel?.name || settings.modelKey;
    select.replaceChildren(remembered);
    select.value = settings.modelKey;
    return;
  }
  const modelKey = forecastModels.some((model) => model.key === settings.modelKey)
    ? settings.modelKey : "base";
  select.value = modelKey;
  if (symbol && modelKey !== settings.modelKey) {
    saveForecastSettings(symbol, { ...settings, modelKey });
  }
}

function restoreSavedForecast(targetSymbol) {
  if (!forecastOptionsLoaded) return;
  if (restoredForecastSymbols.has(targetSymbol)) return;
  const stored = forecastSettings(targetSymbol);
  const settings = forecastModels.some((model) => model.key === stored.modelKey)
    ? stored : { ...stored, modelKey: "base" };
  saveForecastSettings(targetSymbol, settings);
  restoredForecastSymbols.add(targetSymbol);
  if (targetSymbol === symbol) applyForecastSettingsToControls();
  fetch(
    `http://${API_HOST}/api/forecast/${encodeURIComponent(targetSymbol)}?window=${settings.window}&pred_len=${settings.predLen}&model_key=${encodeURIComponent(settings.modelKey)}&modes=${encodeURIComponent(settings.modes.join(","))}&sample_count=${settings.sampleCount}`,
    { method: "POST" },
  ).then((response) => {
    if (!response.ok) restoredForecastSymbols.delete(targetSymbol);
  }).catch(() => restoredForecastSymbols.delete(targetSymbol));
}

async function requestSubscription(targetSymbol) {
  const stored = forecastSettings(targetSymbol);
  const settings = forecastModels.some((model) => model.key === stored.modelKey)
    ? stored : { ...stored, modelKey: "base" };
  saveForecastSettings(targetSymbol, settings);
  const response = await fetch(
    `http://${API_HOST}/api/subscriptions/${encodeURIComponent(targetSymbol)}?window=${settings.window}&pred_len=${settings.predLen}&model_key=${encodeURIComponent(settings.modelKey)}&modes=${encodeURIComponent(settings.modes.join(","))}&sample_count=${settings.sampleCount}`,
    { method: "POST" },
  );
  let result = {};
  try { result = await response.json(); } catch { /* use generic error */ }
  if (!response.ok) throw new Error(result.detail || `无法订阅 ${targetSymbol}`);
  // The backend now owns these preferences before it emits history_snapshot,
  // so the snapshot-triggered first inference already uses the remembered model.
  restoredForecastSymbols.add(targetSymbol);
  return result;
}

async function ensureActivated(targetSymbol) {
  if (activatedSymbols.has(targetSymbol)) return;
  if (activationPromises.has(targetSymbol)) {
    await activationPromises.get(targetSymbol);
    return;
  }
  const activation = requestSubscription(targetSymbol)
    .then(() => activatedSymbols.add(targetSymbol))
    .finally(() => activationPromises.delete(targetSymbol));
  activationPromises.set(targetSymbol, activation);
  await activation;
}

async function requestUnsubscription(targetSymbol) {
  // Closing a WebSocket and processing its disconnect happen asynchronously.
  // Retry a short-lived 409 instead of force-stopping another active viewer.
  for (let attempt = 0; attempt < 6; attempt += 1) {
    const response = await fetch(
      `http://${API_HOST}/api/subscriptions/${encodeURIComponent(targetSymbol)}`,
      { method: "DELETE" },
    );
    let result = {};
    try { result = await response.json(); } catch { /* use generic error */ }
    if (response.ok) return result;
    if (response.status === 409 && attempt < 5) {
      await new Promise((resolve) => setTimeout(resolve, 120));
      continue;
    }
    throw new Error(result.detail || `无法停止监听 ${targetSymbol}`);
  }
  throw new Error(`无法停止监听 ${targetSymbol}`);
}

function connect(generation) {
  clearTimeout(reconnectTimer);
  setConnection("connecting", "正在连接股票池行情");
  const wsProtocol = window.location.protocol === "https:" ? "wss" : "ws";
  const nextSocket = new WebSocket(`${wsProtocol}://${API_HOST}/ws/market`);
  socket = nextSocket;

  nextSocket.addEventListener("open", () => {
    if (generation !== connectionGeneration) return nextSocket.close();
    reconnectAttempt = 0;
    setConnection(
      "online",
      activatedSymbols.has(symbol)
        ? `${symbol} 已连接 · 股票池后台监听`
        : "股票池已连接 · 正在自动订阅",
    );
    nextSocket.send("ready");
    // A fresh page and every reconnect subscribe the complete persisted pool.
    // Clicking a stock only changes the visible chart; it no longer starts data flow.
    const restoreSymbols = [...new Set([
      ...stockPool,
      ...(symbol && !stockPool.includes(symbol) ? [symbol] : []),
    ])];
    Promise.allSettled(restoreSymbols.map((code) => requestSubscription(code)))
      .then((results) => {
        results.forEach((result, index) => {
          if (result.status === "fulfilled") activatedSymbols.add(restoreSymbols[index]);
        });
        const failed = results.filter((result) => result.status === "rejected").length;
        if (failed) {
          setFeedback(`${failed} 只股票后台订阅失败`, "error");
        } else {
          setConnection(
            "online",
            restoreSymbols.length
              ? `股票池 ${restoreSymbols.length} 只股票后台监听中`
              : "股票池为空 · WebSocket 已连接",
          );
        }
      });
  });

  nextSocket.addEventListener("message", ({ data }) => {
    if (generation !== connectionGeneration) return;
    try {
      const event = JSON.parse(data);
      rememberEvent(event);
      if (event.symbol !== symbol) return;
      if (event.type === "history_snapshot") applyHistory(event);
      if (event.type === "bar_update" || event.type === "bar_closed") {
        applyRealtime(event);
        if (event.type === "bar_closed") {
          renderComparison();
          renderStatistics();
          renderSessionTrends();
        }
      }
      if (event.type === "forecast") {
        applyForecast(event);
        renderStatistics();
      }
      if (event.type === "forecast_status") applyForecastStatus(event);
    } catch (error) {
      console.error("无法处理行情事件", error);
    }
  });

  nextSocket.addEventListener("close", () => {
    if (generation !== connectionGeneration) return;
    setConnection("offline", "股票池连接中断，正在重试");
    const delay = Math.min(1000 * 2 ** reconnectAttempt, 10000);
    reconnectAttempt += 1;
    reconnectTimer = setTimeout(() => connect(generation), delay);
  });
  nextSocket.addEventListener("error", () => nextSocket.close());
}

function replaceConnection() {
  connectionGeneration += 1;
  const generation = connectionGeneration;
  clearTimeout(reconnectTimer);
  if (socket) {
    socket.close();
    socket = null;
  }
  connect(generation);
}

async function switchSymbol(targetSymbol) {
  const normalized = targetSymbol.trim().toUpperCase();
  if (!SYMBOL_PATTERN.test(normalized)) {
    setFeedback("代码格式应类似 600519.SH", "error");
    return false;
  }
  if (
    normalized === symbol
    && activatedSymbols.has(normalized)
    && socket?.readyState === WebSocket.OPEN
  ) return true;

  setFeedback(`正在订阅 ${normalized}…`, "loading");
  try {
    await ensureActivated(normalized);
    symbol = normalized;
    if (!stockPool.includes(symbol)) {
      stockPool.push(symbol);
      persistStockPool();
    }
    const url = new URL(window.location.href);
    url.searchParams.set("symbol", symbol);
    window.history.replaceState({}, "", url);
    renderStockPool();
    renderSymbolView();
    setConnection("online", `${symbol} 已连接 · 股票池后台监听`);
    setFeedback(`${symbol} 已加入股票池`, "success");
    return true;
  } catch (error) {
    setFeedback(error.message || "订阅失败", "error");
    return false;
  }
}

async function removeStock(targetSymbol) {
  const removingActive = targetSymbol === symbol;
  const fallback = stockPool.find((code) => code !== targetSymbol);
  if (removingActive && fallback) {
    const switched = await switchSymbol(fallback);
    if (!switched) return;
  }

  setFeedback(`正在停止监听 ${targetSymbol}…`, "loading");
  try {
    await requestUnsubscription(targetSymbol);
    stockPool = stockPool.filter((code) => code !== targetSymbol);
    symbolStates.delete(targetSymbol);
    activatedSymbols.delete(targetSymbol);
    activationPromises.delete(targetSymbol);
    forgetForecastSettings(targetSymbol);
    delete statsStore[targetSymbol];
    delete sessionTrendStore[targetSymbol];
    persistStatsStore();
    persistSessionTrendStore();
    persistStockPool();
    if (removingActive && !fallback) {
      symbol = "";
      const url = new URL(window.location.href);
      url.searchParams.delete("symbol");
      window.history.replaceState({}, "", url);
      renderSymbolView();
      setConnection("online", "股票池为空 · WebSocket 已连接");
    }
    renderStockPool();
    setFeedback(`${targetSymbol} 已移除并停止监听`, "success");
  } catch (error) {
    setFeedback(error.message || "移除失败", "error");
  }
}

element("add-stock-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const input = element("symbol-input");
  const button = element("add-symbol");
  const nextSymbol = input.value.trim().toUpperCase();
  input.classList.toggle("invalid", !SYMBOL_PATTERN.test(nextSymbol));
  if (!SYMBOL_PATTERN.test(nextSymbol)) {
    setFeedback("代码格式应类似 600519.SH", "error");
    input.focus();
    return;
  }
  button.disabled = true;
  button.textContent = "订阅中…";
  await switchSymbol(nextSymbol);
  button.disabled = false;
  button.innerHTML = "<span>＋</span> 添加股票";
  if (symbol === nextSymbol) input.value = "";
});

element("forecast-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  const button = element("forecast-button");
  button.disabled = true;
  try {
    await requestForecast();
  } catch (error) {
    element("forecast-status").textContent = error.message || "预测请求失败";
    element("forecast-status").dataset.state = "error";
  } finally {
    button.disabled = false;
  }
});

for (const id of ["forecast-window", "forecast-length", "sample-count", "forecast-model", "mode-greedy", "mode-median", "mode-mean"]) {
  element(id).addEventListener("change", async () => {
    if (!symbol) return;
    if (id.startsWith("mode-") && !selectedModesFromControls().length) {
      element(id).checked = true;
      setFeedback("至少保留一种预测模式", "error");
      return;
    }
    const modes = selectedModesFromControls();
    saveForecastSettings(symbol, {
      window: Number(element("forecast-window").value),
      predLen: Number(element("forecast-length").value),
      modelKey: element("forecast-model").value || "base",
      sampleCount: Number(element("sample-count").value),
      modes,
    });
    renderStatistics();
    if (id.startsWith("mode-")) {
      const modeKey = id.replace("mode-", "");
      if (!element(id).checked) {
        clearSessionMode(symbol, modeKey);
        stateFor(symbol).forecastModesByTimestamp[modeKey].clear();
      }
      renderSessionTrends();
      try {
        await requestForecast();
      } catch (error) {
        setFeedback(error.message || "无法更新预测模式", "error");
      }
    }
  });
}

persistStockPool();
renderStockPool();
resetComparison();
resetMarketView();
loadForecastOptions().finally(() => replaceConnection());
