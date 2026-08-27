//! Terminal user interface.

use std::collections::{BTreeMap, BTreeSet};
use std::sync::{Arc, RwLock};
use std::{io::stdout, process::Command, time::Instant};
use std::{sync::mpsc, time::Duration};

use ratatui::crossterm::{
  ExecutableCommand,
  event::{self, KeyCode, KeyModifiers},
  terminal,
};
use ratatui::{prelude::*, widgets::*};

use crate::config::{Config, RatioMode, TUI_MAX_MS, TUI_MIN_MS, ViewType};
use macmon::{
  CpuCoreMetrics, FanMetric, MemMetrics, Metrics, Sampler, SocInfo,
  sources::{BatteryStatus, get_battery_status},
};

type WithError<T> = Result<T, Box<dyn std::error::Error>>;

const GB: u64 = 1024 * 1024 * 1024;
const MAX_SPARKLINE: usize = 128;
const MAX_TEMPS: usize = 8;
const BATTERY_LOADS_W: [u64; 3] = [20, 50, 99];
const EFFECTIVE_FULL_HOLD: Duration = Duration::from_secs(5 * 60);

// MARK: Term utils

fn enter_term() -> Terminal<impl Backend> {
  std::panic::set_hook(Box::new(|info| {
    leave_term();
    eprintln!("{}", info);
  }));

  terminal::enable_raw_mode().unwrap();
  stdout().execute(terminal::EnterAlternateScreen).unwrap();

  let term = CrosstermBackend::new(std::io::stdout());
  Terminal::new(term).unwrap()
}

fn leave_term() {
  terminal::disable_raw_mode().unwrap();
  stdout().execute(terminal::LeaveAlternateScreen).unwrap();
}

// MARK: Storage

#[derive(Debug, Default, Clone)]
struct RatioSeries {
  items: Vec<u64>, // Recent percentages (0..=100), newest first.
  ratio: f64,      // Latest ratio (0.0..=1.0).
}

#[derive(Debug, Default, Clone, Copy)]
struct FreqSample {
  freq_mhz: u64,
  scaled_ratio: f64,
  active_ratio: f64,
}

impl FreqSample {
  fn new(freq_mhz: u32, scaled_ratio: f32, active_ratio: f32) -> Self {
    Self {
      freq_mhz: freq_mhz as u64,
      scaled_ratio: scaled_ratio as f64,
      active_ratio: active_ratio as f64,
    }
  }

  fn from_core(core: &CpuCoreMetrics) -> Self {
    Self::new(core.freq_mhz, core.scaled_ratio, core.active_ratio)
  }
}

impl RatioSeries {
  fn push(&mut self, ratio: f64) {
    self.items.insert(0, (ratio * 100.0) as u64);
    self.items.truncate(MAX_SPARKLINE);
    self.ratio = ratio;
  }
}

/// One frequency with parallel scaled and active ratio histories.
#[derive(Debug, Default, Clone)]
struct FreqStore {
  freq_mhz: u64,
  scaled: RatioSeries,
  active: RatioSeries,
}

impl FreqStore {
  fn push(&mut self, sample: FreqSample) {
    self.freq_mhz = sample.freq_mhz;
    self.scaled.push(sample.scaled_ratio);
    self.active.push(sample.active_ratio);
  }

  fn ratio(&self, mode: RatioMode) -> &RatioSeries {
    match mode {
      RatioMode::Scaled => &self.scaled,
      RatioMode::Active => &self.active,
    }
  }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
struct CoreId {
  die_id: usize,
  core_id: usize,
}

impl From<&CpuCoreMetrics> for CoreId {
  fn from(core: &CpuCoreMetrics) -> Self {
    Self { die_id: core.die_id, core_id: core.core_id }
  }
}

#[derive(Debug, Default)]
struct CpuFreqStore {
  aggregate: FreqStore,
  cores: BTreeMap<CoreId, FreqStore>,
}

impl CpuFreqStore {
  fn push(&mut self, aggregate: FreqSample, cores: &[CpuCoreMetrics]) {
    self.aggregate.push(aggregate);

    let mut seen = BTreeSet::new();
    for core in cores {
      let id = CoreId::from(core);
      self.cores.entry(id).or_default().push(FreqSample::from_core(core));
      seen.insert(id);
    }

    for (id, store) in &mut self.cores {
      if !seen.contains(id) {
        store.push(FreqSample::default());
      }
    }
  }

  fn has_multiple_dies(&self) -> bool {
    let Some(first) = self.cores.keys().next() else { return false };
    self.cores.keys().any(|id| id.die_id != first.die_id)
  }
}

#[derive(Debug, Default)]
struct PowerStore {
  items: Vec<u64>,
  top_value: f64,
  max_value: f64,
  avg_value: f64,
}

impl PowerStore {
  fn push(&mut self, value: f64) {
    let was_top = if !self.items.is_empty() { self.items[0] as f64 / 1000.0 } else { 0.0 };

    self.items.insert(0, (value * 1000.0) as u64);
    self.items.truncate(MAX_SPARKLINE);

    self.top_value = avg2(was_top, value);
    self.avg_value = self.items.iter().sum::<u64>() as f64 / self.items.len() as f64 / 1000.0;
    self.max_value = self.items.iter().max().map_or(0, |v| *v) as f64 / 1000.0;
  }
}

#[derive(Debug, Default)]
struct MemoryStore {
  items: Vec<u64>,
  swap_items: Vec<u64>,
  ram_usage: u64,
  ram_total: u64,
  swap_usage: u64,
  swap_total: u64,
  max_ram: u64,
  max_swap: u64,
}

impl MemoryStore {
  fn push(&mut self, value: MemMetrics) {
    self.items.insert(0, value.ram_usage);
    self.items.truncate(MAX_SPARKLINE);

    self.swap_items.insert(0, value.swap_usage);
    self.swap_items.truncate(MAX_SPARKLINE);

    self.ram_usage = value.ram_usage;
    self.ram_total = value.ram_total;
    self.swap_usage = value.swap_usage;
    self.swap_total = value.swap_total;
    self.max_ram = self.items.iter().max().map_or(0, |v| *v);
    self.max_swap = self.swap_items.iter().max().map_or(0, |v| *v);
  }
}

#[derive(Debug, Default)]
struct TempStore {
  items: Vec<f32>,
}

impl TempStore {
  fn last(&self) -> f32 {
    *self.items.first().unwrap_or(&0.0)
  }

  fn push(&mut self, value: f32) {
    // https://www.tunabellysoftware.com/blog/files/tg-pro-apple-silicon-m3-series-support.html
    // https://github.com/vladkens/macmon/issues/12
    let value = if value == 0.0 { self.trend_ema(0.8) } else { value };
    if value == 0.0 {
      return; // skip if not sensor available
    }

    self.items.insert(0, value);
    self.items.truncate(MAX_TEMPS);
  }

  // https://en.wikipedia.org/wiki/Exponential_smoothing
  fn trend_ema(&self, alpha: f32) -> f32 {
    if self.items.len() < 2 {
      return 0.0;
    }

    // starts from most recent value, so need to be reversed
    let mut iter = self.items.iter().rev();
    let mut ema = *iter.next().unwrap_or(&0.0);

    for &item in iter {
      ema = alpha * item + (1.0 - alpha) * ema;
    }

    ema
  }
}

#[derive(Debug, Default)]
struct FanStore {
  items: Vec<FanMetric>,
}

impl FanStore {
  fn push(&mut self, value: Vec<FanMetric>) {
    self.items = value;
  }

  fn label(&self) -> String {
    match self.items.as_slice() {
      [] => "".to_string(),
      [fan] => format!("Fan {} RPM", fan.rpm),
      fans => {
        let values = fans.iter().map(|fan| fan.rpm.to_string()).collect::<Vec<_>>().join("/");
        format!("Fans {values} RPM")
      }
    }
  }
}

fn bar_set() -> symbols::bar::Set<'static> {
  match std::env::var("TERM_PROGRAM").as_deref() {
    Ok("Apple_Terminal") => symbols::bar::THREE_LEVELS,
    _ => symbols::bar::NINE_LEVELS,
  }
}

// MARK: Components

fn h_stack(area: Rect) -> (Rect, Rect) {
  let ha = Layout::default()
    .direction(Direction::Horizontal)
    .constraints([Constraint::Fill(1), Constraint::Fill(1)].as_ref())
    .split(area);

  (ha[0], ha[1])
}

// MARK: Threads

enum Event {
  Update(Box<Metrics>, Option<BatteryStatus>),
  ChangeColor,
  ChangeView,
  TogglePerCore,
  ToggleRatioMode,
  ToggleHelp,
  IncInterval,
  DecInterval,
  Tick,
  Quit,
}

fn handle_key_event(key: &event::KeyEvent, tx: &mpsc::Sender<Event>) -> WithError<()> {
  match key.code {
    KeyCode::Char('q') => Ok(tx.send(Event::Quit)?),
    KeyCode::Char('c') if key.modifiers == KeyModifiers::CONTROL => Ok(tx.send(Event::Quit)?),
    KeyCode::Char('c') => Ok(tx.send(Event::ChangeColor)?),
    KeyCode::Char('v') => Ok(tx.send(Event::ChangeView)?),
    KeyCode::Char('d') => Ok(tx.send(Event::TogglePerCore)?),
    KeyCode::Char('r') => Ok(tx.send(Event::ToggleRatioMode)?),
    KeyCode::Char('?') => Ok(tx.send(Event::ToggleHelp)?),
    KeyCode::Char('+') => Ok(tx.send(Event::IncInterval)?),
    KeyCode::Char('=') => Ok(tx.send(Event::IncInterval)?), // fallback to press without shift
    KeyCode::Char('-') => Ok(tx.send(Event::DecInterval)?),
    _ => Ok(()),
  }
}

fn run_inputs_thread(tx: mpsc::Sender<Event>, tick: u64) {
  let tick_rate = Duration::from_millis(tick);

  std::thread::spawn(move || {
    let mut last_tick = Instant::now();

    loop {
      if event::poll(Duration::from_millis(tick)).unwrap() {
        match event::read().unwrap() {
          event::Event::Key(key) => handle_key_event(&key, &tx).unwrap(),
          _ => {}
        };
      }

      if last_tick.elapsed() >= tick_rate {
        tx.send(Event::Tick).unwrap();
        last_tick = Instant::now();
      }
    }
  });
}

fn run_sampler_thread(tx: mpsc::Sender<Event>, msec: Arc<RwLock<u32>>) {
  std::thread::spawn(move || {
    let mut sampler = Sampler::new().unwrap();

    // Send initial metrics
    tx.send(Event::Update(Box::new(sampler.get_metrics(100).unwrap()), get_battery_status()))
      .unwrap();

    loop {
      let msec = (*msec.read().unwrap()).max(TUI_MIN_MS);
      tx.send(Event::Update(Box::new(sampler.get_metrics(msec).unwrap()), get_battery_status()))
        .unwrap();
    }
  });
}

// get average of two values, used to smooth out metrics
// see: https://github.com/vladkens/macmon/issues/10
fn avg2<T: num_traits::Float>(a: T, b: T) -> T {
  if a == T::zero() { b } else { (a + b) / T::from(2.0).unwrap() }
}

fn ratio(value: f64, total: f64) -> f64 {
  if total == 0.0 { 0.0 } else { value / total }
}

fn battery_bar(capacity: u8) -> String {
  let set = symbols::block::NINE_LEVELS;
  let levels = [
    set.empty,
    set.one_eighth,
    set.one_quarter,
    set.three_eighths,
    set.half,
    set.five_eighths,
    set.three_quarters,
    set.seven_eighths,
  ];
  let eighths = battery_steps(capacity);
  let full = eighths / 8;
  let partial = eighths % 8;

  format!(
    "▕{}{}{}{}",
    set.full.repeat(full),
    if full < 4 { levels[partial] } else { "" },
    set.empty.repeat(4usize.saturating_sub(full + usize::from(full < 4))),
    if capacity < 100 { "▏" } else { "" }
  )
}

fn battery_steps(capacity: u8) -> usize {
  usize::from(capacity.min(100)) * 32 / 100
}

fn battery_color(status: BatteryStatus, primary: Color) -> Color {
  if status.on_ac_power && status.is_charging {
    return Color::Green;
  }
  match status.capacity {
    0..=25 => Color::Red,
    26..50 => Color::Yellow,
    _ => primary,
  }
}

fn battery_label(status: BatteryStatus, primary: Color, runtime: &str) -> Line<'static> {
  let color = battery_color(status, primary);
  let mut spans = vec![Span::styled(format!(" {}", battery_bar(status.capacity)), color)];
  if status.capacity <= 15 || status.capacity < 100 && status.is_charging {
    spans.push(Span::styled(format!(" {}%", status.capacity), primary));
  }
  if status.is_charging || status.on_ac_power && status.input_power_mw.is_some() {
    spans.push(Span::styled(" ⚡", color));
  }
  if !runtime.is_empty() {
    spans.push(Span::styled(format!(" {runtime}"), primary));
  }
  spans.push(Span::raw(" "));
  Line::from(spans)
}

fn battery_runtime_key(status: BatteryStatus) -> Option<(usize, Option<u8>)> {
  (!status.on_ac_power)
    .then(|| (battery_steps(status.capacity), (status.capacity <= 15).then_some(status.capacity)))
}

fn format_minutes(minutes: u64) -> String {
  if minutes >= 60 {
    format!("{}h{:02}m", minutes / 60, minutes % 60)
  } else {
    format!("{minutes}m")
  }
}

fn format_runtime(energy_mwh: u64, watts: u64) -> String {
  format_minutes(energy_mwh.saturating_mul(60) / (watts * 1000))
}

fn battery_time_label(elapsed: Duration) -> String {
  format!("On battery {}", format_minutes(elapsed.as_secs() / 60))
}

fn pmset_field<'a>(entry: &'a str, key: &str) -> Option<&'a str> {
  let value = entry.split_once(key)?.1.trim_start().strip_prefix('=')?.trim_start();
  value.split(|ch: char| ch == ';' || ch == ',' || ch.is_whitespace()).next()
}

fn parse_charge_limit(output: &str) -> Option<u8> {
  let mut manual = None;
  let mut fallback = None;

  for entry in output.split('}') {
    if pmset_field(entry, "Terminated") != Some("0") {
      continue;
    }
    let Some(limit) = pmset_field(entry, "chargeSocLimitSoc")
      .and_then(|value| value.parse::<u8>().ok())
      .filter(|limit| (80..=100).contains(limit))
    else {
      continue;
    };
    let target = if pmset_field(entry, "chargeSocLimitReason") == Some("manualChargeLimit") {
      &mut manual
    } else {
      &mut fallback
    };
    *target = Some(target.map_or(limit, |current: u8| current.min(limit)));
  }

  manual.or(fallback)
}

fn configured_charge_limit() -> Option<u8> {
  let output = Command::new("/usr/bin/pmset").args(["-g", "battlimit"]).output().ok()?;
  output.status.success().then(|| parse_charge_limit(&String::from_utf8_lossy(&output.stdout)))?
}

fn battery_runtime(energy_mwh: u64) -> String {
  BATTERY_LOADS_W
    .iter()
    .map(|watts| format!("{}@{watts}W", format_runtime(energy_mwh, *watts)))
    .collect::<Vec<_>>()
    .join(" ")
}

fn power_label(status: Option<BatteryStatus>) -> String {
  let Some(status) = status.filter(|status| status.on_ac_power) else { return String::new() };
  let input_watts = status.input_power_mw.map(|milliwatts| milliwatts as f64 / 1000.0);
  match (input_watts, status.adapter_watts) {
    (Some(input), Some(max)) => format!("AC {input:.0}/{max}W"),
    (Some(input), None) => format!("AC {input:.0}W"),
    (None, Some(max)) => format!("AC {max}W"),
    (None, None) => "AC connected".to_string(),
  }
}

// MARK: App

#[derive(Debug, Clone, Copy)]
struct ChargeSession {
  capacity: u8,
  elapsed: Duration,
  target: u8,
  learned_target: bool,
}

#[derive(Debug, Clone, Copy)]
struct StableHold {
  capacity: u8,
  elapsed: Duration,
}

#[derive(Debug, Default)]
struct BatteryTimer {
  elapsed: Duration,
  sample: Option<BatteryStatus>,
  sampled_at: Option<Instant>,
  on_ac_power: Option<bool>,
  charge: Option<ChargeSession>,
  hold: Option<StableHold>,
  learned_target: Option<u8>,
}

impl BatteryTimer {
  fn entering_ac(&self, sample: Option<BatteryStatus>) -> bool {
    sample.is_some_and(|status| status.on_ac_power && self.on_ac_power != Some(true))
  }

  fn elapsed_at(&self, now: Instant) -> Option<Duration> {
    self.sample.filter(|status| !status.on_ac_power).map(|_| {
      self.elapsed + self.sampled_at.map(|at| now.saturating_duration_since(at)).unwrap_or_default()
    })
  }

  fn apply_charge(&mut self, status: BatteryStatus) {
    let Some(mut charge) = self.charge else { return };
    if charge.learned_target
      && (status.capacity > charge.target || status.capacity == charge.target && status.is_charging)
    {
      charge.target = 100;
      charge.learned_target = false;
      self.charge = Some(charge);
    }

    let remaining = if status.capacity >= charge.target || charge.capacity >= charge.target {
      Duration::ZERO
    } else {
      charge.elapsed.mul_f64(
        f64::from(charge.target - status.capacity) / f64::from(charge.target - charge.capacity),
      )
    };
    self.elapsed = self.elapsed.min(remaining);
  }

  fn update_hold(
    &mut self,
    status: BatteryStatus,
    previous: Option<BatteryStatus>,
    delta: Duration,
  ) {
    if !status.charging_paused || !(80..100).contains(&status.capacity) {
      self.hold = None;
      return;
    }

    let stable_delta = previous
      .filter(|previous| previous.charging_paused && previous.capacity == status.capacity)
      .map(|_| delta)
      .unwrap_or_default();
    match &mut self.hold {
      Some(hold) if hold.capacity == status.capacity => hold.elapsed += stable_delta,
      hold => *hold = Some(StableHold { capacity: status.capacity, elapsed: Duration::ZERO }),
    }

    if self.hold.is_some_and(|hold| hold.elapsed >= EFFECTIVE_FULL_HOLD) {
      self.elapsed = Duration::ZERO;
      self.learned_target = Some(status.capacity);
      if let Some(charge) = &mut self.charge {
        charge.target = status.capacity;
        charge.learned_target = true;
      }
    }
  }

  fn update(&mut self, sample: Option<BatteryStatus>, now: Instant, configured_target: Option<u8>) {
    let previous = self.sample;
    let delta = self
      .sampled_at
      .map(|sampled_at| now.saturating_duration_since(sampled_at))
      .unwrap_or_default();
    if previous.is_some_and(|status| !status.on_ac_power) {
      self.elapsed += delta;
    }
    self.sampled_at = Some(now);

    let Some(status) = sample else {
      self.sample = None;
      return;
    };

    if self.entering_ac(sample) {
      let (target, learned_target) =
        match configured_target.filter(|limit| (80..=100).contains(limit)) {
          Some(limit) => (limit, false),
          None => self
            .learned_target
            .filter(|limit| status.capacity <= *limit)
            .map_or((100, false), |limit| (limit, true)),
        };
      self.charge = Some(ChargeSession {
        capacity: status.capacity,
        elapsed: self.elapsed,
        target,
        learned_target,
      });
    }
    self.apply_charge(status);

    if status.on_ac_power {
      self.update_hold(status, previous, delta);
    } else {
      self.charge = None;
      self.hold = None;
    }
    self.on_ac_power = Some(status.on_ac_power);
    self.sample = Some(status);
  }
}

#[derive(Debug, Default)]
pub struct App {
  cfg: Config,

  soc: SocInfo,
  mem: MemoryStore,

  cpu_power: PowerStore,
  gpu_power: PowerStore,
  all_power: PowerStore,
  sys_power: PowerStore,

  cpu_temp: TempStore,
  gpu_temp: TempStore,
  fans: FanStore,
  battery: Option<BatteryStatus>,
  battery_timer: BatteryTimer,
  battery_runtime_key: Option<(usize, Option<u8>)>,
  battery_runtime: String,
  show_help: bool,

  ecpu_freq: CpuFreqStore,
  pcpu_freq: CpuFreqStore,
  igpu_freq: FreqStore,
}

impl App {
  pub fn new() -> WithError<Self> {
    let soc = SocInfo::new()?;
    let cfg = Config::load();
    Ok(Self { cfg, soc, ..Default::default() })
  }

  fn update_metrics(&mut self, data: Metrics) {
    self.cpu_power.push(data.cpu_power as f64);
    self.gpu_power.push(data.gpu_power as f64);
    self.all_power.push(data.all_power as f64);
    self.sys_power.push(data.sys_power as f64);

    let ecpu = FreqSample::new(data.ecpu_freq_mhz, data.ecpu_scaled_ratio, data.ecpu_active_ratio);
    let pcpu = FreqSample::new(data.pcpu_freq_mhz, data.pcpu_scaled_ratio, data.pcpu_active_ratio);
    let igpu = FreqSample::new(data.gpu_freq_mhz, data.gpu_scaled_ratio, data.gpu_active_ratio);

    self.ecpu_freq.push(ecpu, &data.ecpu_cores);
    self.pcpu_freq.push(pcpu, &data.pcpu_cores);
    self.igpu_freq.push(igpu);

    self.cpu_temp.push(data.temp.cpu_temp_avg);
    self.gpu_temp.push(data.temp.gpu_temp_avg);
    self.fans.push(data.fans);

    self.mem.push(data.memory);
  }

  fn update_battery(&mut self, battery: Option<BatteryStatus>) {
    let charge_limit =
      self.battery_timer.entering_ac(battery).then(configured_charge_limit).flatten();
    self.battery_timer.update(battery, Instant::now(), charge_limit);

    let key = battery.and_then(battery_runtime_key);
    if key != self.battery_runtime_key {
      self.battery_runtime = battery
        .filter(|status| !status.on_ac_power)
        .and_then(|status| status.remaining_energy_mwh)
        .map(battery_runtime)
        .unwrap_or_default();
      self.battery_runtime_key = key;
    }
    self.battery = battery;
  }

  fn title_block<'a>(&self, label_l: &str, label_r: &str) -> Block<'a> {
    let mut block = Block::new()
      .borders(Borders::ALL)
      .border_type(BorderType::Rounded)
      .border_style(self.cfg.color)
      // .title_style(Style::default().gray())
      .padding(Padding::ZERO);

    if !label_l.is_empty() {
      block = block.title_top(Line::from(format!(" {label_l} ")));
    }

    if !label_r.is_empty() {
      block = block.title_top(Line::from(format!(" {label_r} ")).alignment(Alignment::Right));
    }

    block
  }

  fn get_power_block<'a>(&self, label: &str, val: &'a PowerStore, temp: f32) -> Sparkline<'a> {
    let label_l = format!(
      "{} {:.2}W ({:.2}, {:.2})",
      // "{} {:.2}W (avg: {:.2}W, max: {:.2}W)",
      // "{} {:.2}W (~{:.2}W ^{:.2}W)",
      label,
      val.top_value,
      val.avg_value,
      val.max_value
    );

    let label_r = if temp > 0.0 { format!("{:.1}°C", temp) } else { "".to_string() };

    Sparkline::default()
      .block(self.title_block(label_l.as_str(), label_r.as_str()))
      .direction(RenderDirection::RightToLeft)
      .data(&val.items)
      .style(self.cfg.color)
      .bar_set(bar_set())
  }

  fn render_freq_block(&self, f: &mut Frame, r: Rect, label: &str, val: &FreqStore) {
    let ratio = val.ratio(self.cfg.ratio_mode);
    let label = format!("{} {:3.0}% @ {:4.0} MHz", label, ratio.ratio * 100.0, val.freq_mhz);
    let block = self.title_block(label.as_str(), "");

    match self.cfg.view_type {
      ViewType::Sparkline => {
        let w = Sparkline::default()
          .block(block)
          .direction(RenderDirection::RightToLeft)
          .data(&ratio.items)
          .max(100)
          .style(self.cfg.color)
          .bar_set(bar_set());
        f.render_widget(w, r);
      }
      ViewType::Gauge => {
        let w = Gauge::default()
          .block(block)
          .gauge_style(self.cfg.color)
          .style(self.cfg.color)
          .label("")
          .ratio(ratio.ratio);
        f.render_widget(w, r);
      }
    }
  }

  fn render_cores(&self, f: &mut Frame, r: Rect, label: &str, val: &CpuFreqStore) {
    if val.cores.is_empty() {
      return;
    }

    let aggregate_ratio = val.aggregate.ratio(self.cfg.ratio_mode);

    let title = format!(
      "{} {:3.0}% @ {:4.0} MHz ({} cores)",
      label,
      aggregate_ratio.ratio * 100.0,
      val.aggregate.freq_mhz,
      val.cores.len()
    );
    let block = self.title_block(title.as_str(), "");
    let inner = block.inner(r);
    f.render_widget(block, r);

    // Create vertical layout for each core
    let constraints: Vec<Constraint> = (0..val.cores.len()).map(|_| Constraint::Fill(1)).collect();

    let core_areas =
      Layout::default().direction(Direction::Vertical).constraints(constraints).split(inner);

    // Render each core
    let show_die = val.has_multiple_dies();
    for (i, (id, core)) in val.cores.iter().enumerate() {
      if i >= core_areas.len() {
        break;
      }

      let core = core.ratio(self.cfg.ratio_mode);
      let core_label = if show_die {
        format!("D{} Core {} {:3.0}%", id.die_id, id.core_id, core.ratio * 100.0)
      } else {
        format!("Core {} {:3.0}%", id.core_id, core.ratio * 100.0)
      };

      match self.cfg.view_type {
        ViewType::Sparkline => {
          let w = Sparkline::default()
            .direction(RenderDirection::RightToLeft)
            .data(&core.items)
            .max(100)
            .style(self.cfg.color)
            .bar_set(bar_set());

          // Add a small label for the core
          let label_len = core_label.len();
          let label_span = Span::styled(core_label, Style::default().fg(self.cfg.color));
          let mut area = core_areas[i];

          // Render core label at the start
          if area.width > label_len as u16 {
            let label_area = Rect { x: area.x, y: area.y, width: label_len as u16 + 1, height: 1 };
            f.render_widget(Paragraph::new(label_span), label_area);
            area.x += label_len as u16 + 1;
            area.width = area.width.saturating_sub(label_len as u16 + 1);
          }

          f.render_widget(w, area);
        }
        ViewType::Gauge => {
          let w = Gauge::default()
            .gauge_style(self.cfg.color)
            .style(self.cfg.color)
            .label(core_label)
            .ratio(core.ratio);
          f.render_widget(w, core_areas[i]);
        }
      }
    }
  }

  fn render_mem_block(&self, f: &mut Frame, r: Rect, val: &MemoryStore) {
    let ram_usage_gb = val.ram_usage as f64 / GB as f64;
    let ram_total_gb = val.ram_total as f64 / GB as f64;

    let swap_usage_gb = val.swap_usage as f64 / GB as f64;
    let swap_total_gb = val.swap_total as f64 / GB as f64;

    let ram_pct = ratio(ram_usage_gb, ram_total_gb) * 100.0;
    let label_l = format!("RAM {:4.2} / {:4.1} GB ({:.1}%)", ram_usage_gb, ram_total_gb, ram_pct);
    let label_r = if val.swap_total > 0 {
      format!("SWAP {:.2} / {:.1} GB", swap_usage_gb, swap_total_gb)
    } else {
      String::new()
    };

    let block = self.title_block(label_l.as_str(), label_r.as_str());
    match self.cfg.view_type {
      ViewType::Sparkline => {
        let w = Sparkline::default()
          .block(block)
          .direction(RenderDirection::RightToLeft)
          .data(&val.items)
          .max(val.ram_total)
          .style(self.cfg.color)
          .bar_set(bar_set());
        f.render_widget(w, r);
      }
      ViewType::Gauge => {
        let w = Gauge::default()
          .block(block)
          .gauge_style(self.cfg.color)
          .style(self.cfg.color)
          .label("")
          .ratio(ratio(ram_usage_gb, ram_total_gb));
        f.render_widget(w, r);
      }
    }
  }

  fn gauge_label(&self, label: String, ratio: f64) -> Span<'static> {
    let fg = if ratio > 0.5 { Color::Black } else { self.cfg.color };
    Span::styled(label, Style::default().fg(fg))
  }

  fn render_split_mem_block(&self, f: &mut Frame, r: Rect, val: &MemoryStore) {
    let ram_usage_gb = val.ram_usage as f64 / GB as f64;
    let ram_total_gb = val.ram_total as f64 / GB as f64;
    let swap_usage_gb = val.swap_usage as f64 / GB as f64;
    let swap_total_gb = val.swap_total as f64 / GB as f64;

    let title = "Memory";
    let block = self.title_block(title, "");
    let inner = block.inner(r);
    f.render_widget(block, r);

    let constraints = if val.swap_total > 0 {
      vec![Constraint::Fill(1), Constraint::Fill(1)]
    } else {
      vec![Constraint::Fill(1)]
    };
    let sections =
      Layout::default().direction(Direction::Vertical).constraints(constraints).split(inner);

    // RAM section
    let ram_label = format!("RAM {:4.2}/{:4.1} GB", ram_usage_gb, ram_total_gb);
    match self.cfg.view_type {
      ViewType::Sparkline => {
        let w = Sparkline::default()
          .direction(RenderDirection::RightToLeft)
          .data(&val.items)
          .max(val.ram_total)
          .style(self.cfg.color)
          .bar_set(bar_set());

        let label_len = ram_label.len();
        let label_span = Span::styled(ram_label, Style::default().fg(self.cfg.color));
        let mut area = sections[0];

        if area.width > label_len as u16 {
          let label_area = Rect { x: area.x, y: area.y, width: label_len as u16 + 1, height: 1 };
          f.render_widget(Paragraph::new(label_span), label_area);
          area.x += label_len as u16 + 1;
          area.width = area.width.saturating_sub(label_len as u16 + 1);
        }

        f.render_widget(w, area);
      }
      ViewType::Gauge => {
        let ratio = ratio(ram_usage_gb, ram_total_gb);
        let w = Gauge::default()
          .gauge_style(self.cfg.color)
          .style(self.cfg.color)
          .label(self.gauge_label(ram_label, ratio))
          .ratio(ratio);
        f.render_widget(w, sections[0]);
      }
    }

    if val.swap_total == 0 {
      return;
    }

    // SWAP section
    let swap_label = format!("SWAP {:4.2}/{:4.1} GB", swap_usage_gb, swap_total_gb);
    match self.cfg.view_type {
      ViewType::Sparkline => {
        let w = Sparkline::default()
          .direction(RenderDirection::RightToLeft)
          .data(&val.swap_items)
          .max(val.swap_total.max(1)) // Avoid division by zero if no swap
          .style(self.cfg.color)
          .bar_set(bar_set());

        let label_len = swap_label.len();
        let label_span = Span::styled(swap_label, Style::default().fg(self.cfg.color));
        let mut area = sections[1];

        if area.width > label_len as u16 {
          let label_area = Rect { x: area.x, y: area.y, width: label_len as u16 + 1, height: 1 };
          f.render_widget(Paragraph::new(label_span), label_area);
          area.x += label_len as u16 + 1;
          area.width = area.width.saturating_sub(label_len as u16 + 1);
        }

        f.render_widget(w, area);
      }
      ViewType::Gauge => {
        let ratio = ratio(swap_usage_gb, swap_total_gb);
        let w = Gauge::default()
          .gauge_style(self.cfg.color)
          .style(self.cfg.color)
          .label(self.gauge_label(swap_label, ratio))
          .ratio(ratio);
        f.render_widget(w, sections[1]);
      }
    }
  }

  fn render(&mut self, f: &mut Frame) {
    let label_l = format!(
      "{} ({}{}+{}{}+{}GPU {}GB)",
      self.soc.chip_name,
      self.soc.ecpu_cores,
      self.soc.ecpu_label,
      self.soc.pcpu_cores,
      self.soc.pcpu_label,
      self.soc.gpu_cores,
      self.soc.memory_gb,
    );

    let rows = Layout::default()
      .direction(Direction::Vertical)
      .constraints([Constraint::Fill(2), Constraint::Fill(1)].as_ref())
      .split(f.area());

    let brand = format!("{} v{}", env!("CARGO_PKG_NAME"), env!("CARGO_PKG_VERSION"));
    let block = self.title_block(&label_l, &brand);
    let iarea = block.inner(rows[0]);
    f.render_widget(block, rows[0]);

    let iarea = Layout::default()
      .direction(Direction::Vertical)
      .constraints([Constraint::Fill(1), Constraint::Fill(1)].as_ref())
      .split(iarea);

    // 1st row
    let (c1, c2) = h_stack(iarea[0]);
    let ecpu_block_label = format!("{}-CPU", self.soc.ecpu_label);
    let pcpu_block_label = format!("{}-CPU", self.soc.pcpu_label);
    if self.cfg.per_core_view {
      self.render_cores(f, c1, &ecpu_block_label, &self.ecpu_freq);
      self.render_cores(f, c2, &pcpu_block_label, &self.pcpu_freq);
    } else {
      self.render_freq_block(f, c1, &ecpu_block_label, &self.ecpu_freq.aggregate);
      self.render_freq_block(f, c2, &pcpu_block_label, &self.pcpu_freq.aggregate);
    }

    // 2nd row
    let (c1, c2) = h_stack(iarea[1]);
    if self.cfg.per_core_view {
      self.render_split_mem_block(f, c1, &self.mem);
    } else {
      self.render_mem_block(f, c1, &self.mem);
    }
    self.render_freq_block(f, c2, "GPU", &self.igpu_freq);

    // 3rd row
    let label_l = format!(
      "Power: {:.2}W (avg {:.2}W, max {:.2}W)",
      self.all_power.top_value, self.all_power.avg_value, self.all_power.max_value,
    );

    let label_r = [
      self.battery_timer.elapsed_at(Instant::now()).map(battery_time_label).unwrap_or_default(),
      power_label(self.battery),
      self.fans.label(),
    ]
    .into_iter()
    .filter(|label| !label.is_empty())
    .collect::<Vec<_>>()
    .join(" | ");

    let mut block = self.title_block(&label_l, &label_r);
    if let Some(battery) = self.battery {
      block = block.title_bottom(battery_label(battery, self.cfg.color, &self.battery_runtime));
    }
    let block = block.title_bottom(Line::from(" ? ").right_aligned());
    let iarea = block.inner(rows[1]);
    f.render_widget(block, rows[1]);

    let ha = Layout::default()
      .direction(Direction::Horizontal)
      .constraints([Constraint::Fill(1), Constraint::Fill(1), Constraint::Fill(1)].as_ref())
      .split(iarea);

    f.render_widget(self.get_power_block("CPU", &self.cpu_power, self.cpu_temp.last()), ha[0]);
    f.render_widget(self.get_power_block("GPU", &self.gpu_power, self.gpu_temp.last()), ha[1]);
    f.render_widget(self.get_power_block("Total", &self.sys_power, 0.0), ha[2]);

    if self.show_help {
      let help = [
        "?           toggle help",
        "q / Ctrl-C  quit",
        "c           change color",
        "v           change chart type",
        "d           toggle detail view",
        "r           change ratio mode",
        "+ / =       increase interval",
        "-           decrease interval",
      ];
      let area = f.area();
      let width = help.iter().map(|line| line.len()).max().unwrap_or(0) as u16 + 2;
      let popup = Rect::new(
        area.x + area.width.saturating_sub(width.min(area.width)) / 2,
        area.y + area.height.saturating_sub((help.len() as u16 + 2).min(area.height)) / 2,
        width.min(area.width),
        (help.len() as u16 + 2).min(area.height),
      );
      f.render_widget(Clear, popup);
      f.render_widget(
        Paragraph::new(help.join("\n")).block(self.title_block("Help", "")).style(self.cfg.color),
        popup,
      );
    }
  }

  pub fn run_loop(&mut self, interval: Option<u32>) -> WithError<()> {
    // use from arg if provided, otherwise use config restored value
    self.cfg.interval = interval.unwrap_or(self.cfg.interval).clamp(TUI_MIN_MS, TUI_MAX_MS);
    let msec = Arc::new(RwLock::new(self.cfg.interval));

    let (tx, rx) = mpsc::channel::<Event>();
    run_inputs_thread(tx.clone(), 250);
    run_sampler_thread(tx.clone(), msec.clone());

    let mut term = enter_term();

    loop {
      term.draw(|f| self.render(f)).unwrap();

      match rx.recv()? {
        Event::Quit => break,
        Event::Update(data, battery) => {
          self.update_metrics(*data);
          self.update_battery(battery);
        }
        Event::ChangeColor => self.cfg.next_color(),
        Event::ChangeView => self.cfg.next_view_type(),
        Event::TogglePerCore => self.cfg.toggle_per_core_view(),
        Event::ToggleRatioMode => self.cfg.toggle_ratio_mode(),
        Event::ToggleHelp => self.show_help = !self.show_help,
        Event::IncInterval => {
          self.cfg.inc_interval();
          *msec.write().unwrap() = self.cfg.interval;
        }
        Event::DecInterval => {
          self.cfg.dec_interval();
          *msec.write().unwrap() = self.cfg.interval;
        }
        _ => {}
      }
    }

    leave_term();
    Ok(())
  }
}

#[cfg(test)]
mod tests {
  use super::{
    BatteryTimer, battery_bar, battery_color, battery_label, battery_runtime, battery_runtime_key,
    battery_time_label, format_runtime, parse_charge_limit, power_label,
  };
  use macmon::sources::BatteryStatus;
  use ratatui::{style::Color, text::Line};
  use std::time::Duration;

  fn battery(capacity: u8, is_charging: bool, on_ac_power: bool) -> BatteryStatus {
    BatteryStatus {
      capacity,
      is_charging,
      on_ac_power,
      charging_paused: false,
      adapter_watts: None,
      input_power_mw: None,
      remaining_energy_mwh: None,
    }
  }

  #[test]
  fn renders_battery_levels_and_colors() {
    for (capacity, expected) in [
      (0, "▕    ▏"),
      (1, "▕    ▏"),
      (15, "▕▌   ▏"),
      (25, "▕█   ▏"),
      (50, "▕██  ▏"),
      (75, "▕███ ▏"),
      (99, "▕███▉▏"),
      (100, "▕████"),
    ] {
      assert_eq!(battery_bar(capacity), expected);
    }

    assert_eq!(battery_color(battery(25, false, false), Color::Blue), Color::Red);
    assert_eq!(battery_color(battery(26, false, false), Color::Blue), Color::Yellow);
    assert_eq!(battery_color(battery(49, false, false), Color::Blue), Color::Yellow);
    assert_eq!(battery_color(battery(50, false, false), Color::Blue), Color::Blue);
    assert_eq!(battery_color(battery(10, true, true), Color::Blue), Color::Green);

    let base = battery(79, false, false);
    let text =
      |line: Line<'_>| line.spans.iter().map(|span| span.content.as_ref()).collect::<String>();
    assert_eq!(text(battery_label(base, Color::Green, "")), " ▕███▏▏ ");
    let charging = battery_label(battery(79, true, true), Color::Blue, "");
    assert_eq!(text(charging.clone()), " ▕███▏▏ 79% ⚡ ");
    assert_eq!(charging.spans[0].style.fg, Some(Color::Green));
    assert_eq!(charging.spans[1].style.fg, Some(Color::Blue));
    assert_eq!(charging.spans[2].style.fg, Some(Color::Green));

    let low = battery_label(battery(15, true, false), Color::Green, "");
    assert_eq!(text(low.clone()), " ▕▌   ▏ 15% ⚡ ");
    assert_eq!(low.spans[0].style.fg, Some(Color::Red));
    assert_eq!(low.spans[1].style.fg, Some(Color::Green));
    assert_eq!(text(battery_label(battery(16, false, false), Color::Green, "")), " ▕▋   ▏ ");

    let mut powered = battery(15, false, true);
    powered.input_power_mw = Some(10_000);
    assert_eq!(text(battery_label(powered, Color::Green, "")), " ▕▌   ▏ 15% ⚡ ");
    assert_eq!(text(battery_label(battery(100, true, true), Color::Blue, "")), " ▕████ ⚡ ");

    let mut adapter = battery(50, true, true);
    assert_eq!(power_label(Some(adapter)), "AC connected");
    adapter.adapter_watts = Some(65);
    assert_eq!(power_label(Some(adapter)), "AC 65W");
    adapter.input_power_mw = Some(46_355);
    assert_eq!(power_label(Some(adapter)), "AC 46/65W");
    adapter.adapter_watts = None;
    assert_eq!(power_label(Some(adapter)), "AC 46W");
    assert_eq!(power_label(Some(battery(50, false, false))), "");
  }

  #[test]
  fn predicts_runtime_at_visible_battery_granularity() {
    assert_eq!(format_runtime(46_667, 20), "2h20m");
    assert_eq!(battery_runtime(46_667), "2h20m@20W 56m@50W 28m@99W");
    assert_eq!(
      battery_runtime_key(battery(79, false, false)),
      battery_runtime_key(battery(81, false, false))
    );
    assert_ne!(
      battery_runtime_key(battery(81, false, false)),
      battery_runtime_key(battery(82, false, false))
    );
    assert_ne!(
      battery_runtime_key(battery(14, false, false)),
      battery_runtime_key(battery(15, false, false))
    );
    assert_eq!(battery_runtime_key(battery(50, false, true)), None);

    let text =
      |line: Line<'_>| line.spans.iter().map(|span| span.content.as_ref()).collect::<String>();
    assert_eq!(
      text(battery_label(battery(79, false, false), Color::Green, "2h20m@20W")),
      " ▕███▏▏ 2h20m@20W "
    );
  }

  #[test]
  fn scales_battery_time_across_partial_charges() {
    assert_eq!(battery_time_label(Duration::ZERO), "On battery 0m");
    assert_eq!(battery_time_label(Duration::from_secs(83 * 60)), "On battery 1h23m");

    let start = std::time::Instant::now();
    let mut timer = BatteryTimer::default();
    timer.update(Some(battery(80, false, false)), start, None);
    timer.update(Some(battery(80, false, false)), start + Duration::from_secs(60 * 60), None);
    timer.update(Some(battery(80, true, true)), start + Duration::from_secs(60 * 60), Some(100));
    timer.update(Some(battery(90, true, true)), start + Duration::from_secs(70 * 60), None);
    assert_eq!(timer.elapsed, Duration::from_secs(30 * 60));
    assert_eq!(timer.elapsed_at(start + Duration::from_secs(70 * 60)), None);

    timer.update(Some(battery(90, false, false)), start + Duration::from_secs(70 * 60), None);
    assert_eq!(
      timer.elapsed_at(start + Duration::from_secs(85 * 60)),
      Some(Duration::from_secs(45 * 60))
    );
  }

  #[test]
  fn honors_explicit_charge_targets_and_full_charge() {
    let start = std::time::Instant::now();
    let mut timer = BatteryTimer::default();
    timer.update(Some(battery(80, false, false)), start, None);
    timer.update(Some(battery(80, false, false)), start + Duration::from_secs(60 * 60), None);
    timer.update(Some(battery(80, true, true)), start + Duration::from_secs(60 * 60), Some(90));
    timer.update(Some(battery(85, true, true)), start + Duration::from_secs(65 * 60), None);
    assert_eq!(timer.elapsed, Duration::from_secs(30 * 60));
    timer.update(Some(battery(90, false, true)), start + Duration::from_secs(70 * 60), None);
    assert_eq!(timer.elapsed, Duration::ZERO);

    let mut timer = BatteryTimer { elapsed: Duration::from_secs(60 * 60), ..Default::default() };
    timer.update(Some(battery(95, true, true)), start, Some(100));
    timer.update(Some(battery(100, false, true)), start + Duration::from_secs(5 * 60), None);
    assert_eq!(timer.elapsed, Duration::ZERO);
  }

  #[test]
  fn learns_only_stable_effective_full_holds() {
    let start = std::time::Instant::now();
    let mut timer = BatteryTimer { elapsed: Duration::from_secs(60 * 60), ..Default::default() };
    let mut paused = battery(80, false, true);
    paused.charging_paused = true;

    timer.update(Some(paused), start, None);
    timer.update(Some(paused), start + Duration::from_secs(4 * 60 + 59), None);
    assert_eq!(timer.elapsed, Duration::from_secs(60 * 60));
    timer.update(Some(battery(80, true, true)), start + Duration::from_secs(5 * 60), None);
    assert_eq!(timer.learned_target, None);

    timer.update(Some(paused), start + Duration::from_secs(6 * 60), None);
    timer.update(Some(paused), start + Duration::from_secs(11 * 60), None);
    assert_eq!(timer.learned_target, Some(80));
    assert_eq!(timer.elapsed, Duration::ZERO);
  }

  #[test]
  fn charging_past_a_learned_target_uses_full_capacity() {
    let start = std::time::Instant::now();
    let mut timer = BatteryTimer {
      elapsed: Duration::from_secs(60 * 60),
      learned_target: Some(80),
      ..Default::default()
    };
    timer.update(Some(battery(80, true, true)), start, None);
    assert_eq!(timer.elapsed, Duration::from_secs(60 * 60));
    assert_eq!(timer.charge.unwrap().target, 100);
  }

  #[test]
  fn missing_battery_samples_freeze_and_hide_time() {
    let start = std::time::Instant::now();
    let mut timer = BatteryTimer::default();
    timer.update(Some(battery(70, false, false)), start, None);
    timer.update(None, start + Duration::from_secs(10 * 60), None);
    assert_eq!(timer.elapsed_at(start + Duration::from_secs(40 * 60)), None);

    timer.update(Some(battery(69, false, false)), start + Duration::from_secs(40 * 60), None);
    assert_eq!(
      timer.elapsed_at(start + Duration::from_secs(45 * 60)),
      Some(Duration::from_secs(15 * 60))
    );
  }

  #[test]
  fn parses_active_pmset_charge_limits() {
    let limits = r#"Battery level limits:
      ( { chargeSocLimitReason = optimizedCharging; chargeSocLimitSoc = 85; Terminated = 0; },
        { chargeSocLimitReason = manualChargeLimit; chargeSocLimitSoc = 90; Terminated = 0; },
        { chargeSocLimitReason = manualChargeLimit; chargeSocLimitSoc = 80; Terminated = 1; } )"#;
    assert_eq!(parse_charge_limit(limits), Some(90));

    let fallback = r#"( { chargeSocLimitSoc = 95; Terminated = 0; },
      { chargeSocLimitSoc = 85; Terminated = 0; },
      { chargeSocLimitSoc = 70; Terminated = 0; } )"#;
    assert_eq!(parse_charge_limit(fallback), Some(85));
    assert_eq!(parse_charge_limit("No battery level limits set"), None);
  }
}
